import copy
import numpy as np
import re
from fnmatch import fnmatch

import torch
import torch.nn as nn

from .layer import LayerBuilder
from .prune_utils import get_masks

tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaPruner:
    def __init__(self, model, loader, logger, args, passer=None):
        self.model = model
        self.args = args
        self.logger = logger
        self.loader = loader
        self.accprint = lambda x: print(x, acc=True)
        self.dummy_input = passer.dummy_input
        self.test = lambda net: passer.test(loader.test_loader, net, passer.criterion, args, print=False)

        # Set up layers
        self.LEARNABLES = (nn.Conv2d, nn.Conv1d, nn.Linear, nn.MultiheadAttention) # The layers for pruning. Constant! You may need to change it to your need
        self.learnable_layers = self.LEARNABLES # To maintain back-compatibility, will be removed
        print(f'Learnable layer types: {self.LEARNABLES} -- layers of these types will be accounted for in pruning')
        arch = args.arch if hasattr(args, 'arch') else None
        layer_builder = LayerBuilder(model, self.LEARNABLES, arch=arch, dummy_input=self.dummy_input)
        self.layers = layer_builder.layers
        self._max_len_ix = layer_builder._max_len_ix
        self._max_len_name = layer_builder._max_len_name
        self.layer_print_prefix = layer_builder.print_prefix

        # Set up pr for each layer
        from .prune_utils import get_pr_model
        self.raw_pr = get_pr_model(self.layers, args.stage_pr, skip=args.skip_layers, compare_mode=args.compare_mode)
        self.pr = copy.deepcopy(self.raw_pr)

        # Get contrained layers (if any)
        # TODO-@mst: Move to the layer construction init fn
        from .prune_utils import get_constrained_layers
        self.constrained_layers, self.layers = get_constrained_layers(self.layers, args.same_pruned_wg_layers)
        if len(self.constrained_layers):
            print(f'Constrained layers: {self.constrained_layers}')

        # Get skip layers
        self.skip_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, self.LEARNABLES):
                for p in args.skip_layers:
                    if fnmatch(name, p):
                        self.skip_layers += [name]
        print(f'Skip layers: {self.skip_layers}')

        # Number of layers that will be pruned
        self.num_prunable_layers = self._get_prunable_index()

        # Get pruning order for each layer. Useful in layerwise pruning
        self.layerwise_prune_interval = self._get_prune_order()
    def _get_prunable_index(self):
        cnt = -1
        for name, layer in self.layers.items():
            if self.pr[name] > 0:
                cnt += 1
                layer.prunable_index = cnt
            else:
                layer.prunable_index = None
        return cnt + 1
    def _get_prune_order(self):
        r"""Get the prune order for each prunable layer (pr > 0).
        """
        prune_interval = 0
        for name, layer in self.layers.items():
            if self.pr[name] > 0:
                if self.args.prune_schedule in ['simu']:
                    layer.prune_order = 0
                elif re.match('ascend_\d+', self.args.prune_schedule):
                    layer.prune_order = layer.prunable_index
                    prune_interval = int(self.args.prune_schedule.split('_')[1])
                elif re.match('descend_\d+', self.args.prune_schedule):
                    layer.prune_order = self.num_prunable_layers - 1 - layer.prunable_index
                    prune_interval = int(self.args.prune_schedule.split('_')[1])
                else:
                    raise NotImplementedError
            else:
                layer.prune_order = None
        return prune_interval

    def _get_pr_proportionally(self,
                               global_pr: float,
                               proportion: dict,
                               sort_by: str = 'params',
                               skip_layers: list = []):
        self.pr = {}
        skip_layers_ = [x for x in skip_layers]
        prop = {}

        def get_base_ratio():
            layer_complexity = []
            layer_complexity_toprune, keep_factor, keep_factor_dict = [], [], {}
            layer_complexity_tokeep = []
            nonlocal skip_layers_

            for name, module in self.model.named_modules():
                if isinstance(module, self.learnable_layers):
                    # Get proportion
                    for key, value in proportion.items():
                        if fnmatch(name, key):
                            p = value

                    # Get layer complexity (params or flops)
                    if sort_by in ['params']:
                        comp = module.weight.numel()
                    else:
                        raise NotImplementedError
                    layer_complexity += [comp]

                    # Decide if the layer will not be pruned
                    skip = False
                    for s in skip_layers_:
                        if fnmatch(name, s):
                            skip = True
                    if skip:
                        self.pr[name] = 0
                        layer_complexity_tokeep += [comp]
                    else:
                        layer_complexity_toprune += [comp]
                        keep_factor += [p]
                        keep_factor_dict[name] = p

            base_ratio = (np.sum(layer_complexity) * (1 - global_pr) - np.sum(layer_complexity_tokeep)) / \
                         np.dot(layer_complexity_toprune, keep_factor)

            # Get layer-wise pr
            pass_ = True
            for name, module in self.model.named_modules():
                if isinstance(module, self.learnable_layers):
                    if name not in self.pr:
                        sparsity = 1 - base_ratio * keep_factor_dict[name]
                        if sparsity < 0:
                            pass_ = False
                            skip_layers_.append(name)
                            self.pr[name] = 0

            return pass_, base_ratio, keep_factor_dict

        pass_, base_ratio, keep_factor_dict = get_base_ratio()
        while not pass_:
            print(f'get_base_ratio does NOT pass, do it again!')
            pass_, base_ratio, keep_factor_dict = get_base_ratio()

        # Finally, get the pr
        for name, module in self.model.named_modules():
            if isinstance(module, self.learnable_layers):
                if name not in self.pr:
                    sparsity = 1 - base_ratio * keep_factor_dict[name]
                    assert 0 <= sparsity < 1
                    self.pr[name] = sparsity

        # Print
        for name in self.pr:
            print(f'{name} -- {self.pr[name]}')

    def _get_kept_wg(self, align_constrained=False, criterion='mag', sort_mode=None, is_print=True):
        r"""Get kept/pruned weight groups for the model
        TODO-@mst: Should put this in prune_utils.py ?
        """
        # Update args
        if sort_mode is None:
            sort_mode = self.args.pick_pruned

        if hasattr(self.args, 'inherit_pruned') and self.args.inherit_pruned == 'index':
            import os
            assert os.path.exists(self.args.stage_pr)
            ckpt = torch.load(self.args.stage_pr)
            pr = self.raw_pr
            pruned_wg, kept_wg = ckpt['pruned_wg'], ckpt['kept_wg']
            scheme = f'inheriting_existing_pruned_indices'
            if is_print:
                print(f"==> Load base_pr model successfully and inherit its pruned indices: '{self.args.stage_pr}'")
        else:
            # ************************* Core pruning function **************************
            from .prune_utils import pick_pruned_model, get_score_layer
            pr, pruned_wg, kept_wg = pick_pruned_model(self.model, self.layers, self.raw_pr,
                                                       get_score_layer=get_score_layer,
                                                       wg=self.args.wg,
                                                       criterion=criterion,
                                                       compare_mode=self.args.compare_mode,
                                                       sort_mode=sort_mode,
                                                       constrained=self.constrained_layers,
                                                       align_constrained=align_constrained)
            
            scheme = f'{self.args.prune_method, criterion, self.args.pick_pruned}'
            # ***************************************************************************

        # Print
        if is_print:
            print(f'*********** Get pruned wg ***********')
            for name, layer in self.layers.items():
                logtmp = f'{self.layer_print_prefix[name]} -- Got pruned wg by {scheme}, pr {pr[name]}'
                ext = f' -- This is a constrained layer' if name in self.constrained_layers else ''
                print(logtmp + ext)
            print(f'*************************************')
            # TODO-@mst: here, the printed info should be improved

        return pr, pruned_wg, kept_wg

    def _prune_and_build_new_model(self):
        acc1_before, *_ = self.test(self.model)
        from .prune_utils import get_masks, get_kept_filter_channel, replace_module, get_next_bn

        if self.args.wg == 'weight':
            self.masks = get_masks(self.layers, self.pruned_wg)

            # Reinit designated layers
            for name, m in self.model.named_modules():
                if isinstance(m, self.LEARNABLES):
                    reinit = False
                    for rl in self.args.reinit_layers:
                        if fnmatch(name, rl):
                            reinit = True
                            break
                    if reinit:
                        m.reset_parameters()
                        print(f'Layer {name} is reinited when building the new model!')
            return

        new_model = copy.deepcopy(self.model)
        for name, m in self.model.named_modules():
            if isinstance(m, self.LEARNABLES):
                kept_filter, kept_chl = get_kept_filter_channel(self.layers, name, pr=self.pr, kept_wg=self.kept_wg,
                                                                wg=self.args.wg)
                
                # decide if renit the current layer
                reinit = False
                for rl in self.args.reinit_layers:
                    if fnmatch(name, rl):
                        reinit = True
                        break

                # get number of channels (can be manually assigned)
                num_chl = self.args.layer_chl[name] if name in self.args.layer_chl else len(kept_chl)

                # copy weight and bias
                bias = False if isinstance(m.bias, type(None)) else True
                if isinstance(m, nn.Conv2d):
                    new_layer = nn.Conv2d(num_chl, len(kept_filter), m.kernel_size,
                                          m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                    if not reinit:
                        kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]

                elif isinstance(m, nn.Linear):
                    kept_weights = m.weight.data[kept_filter][:, kept_chl]
                    new_layer = nn.Linear(in_features=len(kept_chl), out_features=len(kept_filter), bias=bias).cuda()

                if not reinit:
                    new_layer.weight.data.copy_(kept_weights)  # load weights into the new module
                    if bias:
                        kept_bias = m.bias.data[kept_filter]
                        new_layer.bias.data.copy_(kept_bias)
                else:
                    print(f'Layer {name} is reinited when building the new model!')

                # load the new conv
                replace_module(new_model, name, new_layer)

                # get the corresponding bn (if any) for later use
                next_bn = get_next_bn(self.model, name)

            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                # print()
                # print(len(kept_filter), m.weight.data.shape, m.bias.data.shape)
                # print(kept_filter, type(kept_filter)) # For debug
                new_bn = nn.BatchNorm2d(len(kept_filter), eps=m.eps, momentum=m.momentum,
                                        affine=m.affine, track_running_stats=m.track_running_stats).cuda()

                # copy bn weight and bias
                weight = m.weight.data[kept_filter]
                new_bn.weight.data.copy_(weight)
                bias = m.bias.data[kept_filter]
                new_bn.bias.data.copy_(bias)

                # copy bn running stats
                new_bn.running_mean.data.copy_(m.running_mean[kept_filter])
                new_bn.running_var.data.copy_(m.running_var[kept_filter])
                new_bn.num_batches_tracked.data.copy_(m.num_batches_tracked)

                # load the new bn
                replace_module(new_model, name, new_bn)

        self.model = new_model

        # print the layer shape of pruned model
        arch = self.args.arch if hasattr(self.args, 'arch') else None
        LayerBuilder(self.model, self.LEARNABLES, arch=arch, dummy_input=self.dummy_input)

        acc1_after, *_ = self.test(self.model)
        self.accprint("Acc1 %.8f -- Before _prune_and_build_new_model" % (acc1_before))
        self.accprint("Acc1 %.8f -- After  _prune_and_build_new_model" % (acc1_after))

        return new_model
    
    def _check_weight(self):
        r"""Check weights and their masks
        """
        # Select layers to print
        pruned_layers = [n for n in self.layers if self.pr[n] > 0]
        num = len(pruned_layers)
        selected_ix = [0, int(num * 0.25), int(num * 0.5), int(num * 0.75), num - 1]
        selected_layers = [pruned_layers[i] for i in selected_ix]

        masks = get_masks(self.layers, self.pruned_wg, self.args.wg)
        num_print = 10
        for name, m in self.model.named_modules():
            if name in selected_layers:
                weight_ = m.weight.data.flatten()
                mask_ = masks[name].flatten()
                np.random.seed(0)
                ix = np.random.choice(len(mask_), num_print)
                wstr = ' '.join([f'{x:.5f}({int(y.item())})'.rjust(11) for x, y in zip(weight_[ix], mask_[ix])])
                print(f'{self.layers[name].index} {name} weight(mask): {wstr}')
    
    def _check_bn(self):
        # Select layers to print
        pruned_layers = [n for n in self.layers if self.pr[n] > 0]
        num = len(pruned_layers)
        selected_ix = [0, int(num * 0.25), int(num * 0.5), int(num * 0.75), num - 1]
        selected_layers = [pruned_layers[i] for i in selected_ix]

        assert self.args.wg == 'filter'
        num_print = 10
        all_layers = [n for n, _ in self.model.named_modules()]  # Conv, ReLU, BN, FC, etc.
        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # Get the associating conv layer of this BN layer
                ix = all_layers.index(name)
                for k in range(ix - 1, -1, -1):
                    if all_layers[k] in self.layers:
                        last_conv = all_layers[k]
                        break
                if last_conv not in selected_layers: continue
                mask_ = [0] * m.weight.data.size(0)
                for i in self.kept_wg[last_conv]:
                    mask_[i] = 1
                wstr = ' '.join([f'{x:.3f}({y})'.rjust(9) for x, y in zip(m.weight.data[:num_print], mask_[:num_print])])
                bstr = ' '.join([f'{x:.3f}({y})'.rjust(9) for x, y in zip(m.bias.data[:num_print], mask_[:num_print])])
                print(f'{self.layers[last_conv].index} {last_conv} BN weight: {wstr}')
                print(f'{self.layers[last_conv].index} {last_conv} BN bias  : {bstr}')

    def _check_mask_overlap_with_MP(self):
        # Select layers to print
        pruned_layers = [n for n in self.layers if self.pr[n] > 0]
        num = len(pruned_layers)
        selected_ix = [0, int(num * 0.25), int(num * 0.5), int(num * 0.75), num - 1]
        selected_layers = [pruned_layers[i] for i in selected_ix]

        _, pruned_wg_MP_temp, kept_wg_MP_temp = self._get_kept_wg(self.args.align_constrained, sort_mode='min', is_print=False)
        print("-" * 20 + ' Check mask overlap ' + "-" * 20)
        for name, layer in self.layers.items():
            if name in selected_layers:
                overlap = [x for x in pruned_wg_MP_temp[name] if x in self.pruned_wg[name]]
                r = len(overlap) / len(pruned_wg_MP_temp[name])
                print(f'{layer.index} {name} -- Overlapped pruned_wg: {len(overlap)} ({r * 100:.2f}%) PR: {self.pr[name]}')
        print("-" * 20 + '--------------------' + "-" * 20)