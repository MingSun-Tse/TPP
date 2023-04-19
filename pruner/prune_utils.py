import torch, torch.nn as nn
from collections import OrderedDict
from fnmatch import fnmatch
import math, numpy as np, copy
import re
from scipy.linalg import circulant
import sys

tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)

def get_pr_layer(base_pr, layer_name, layer_index, skip=[], compare_mode='local'):
    r"""'base_pr' example: '[0-4:0.5, 5:0.6, 8-10:0.2]', 6, 7 not mentioned, default value is 0
    """
    if compare_mode in ['global']:
        pr = 1e-20 # A small positive value to indicate this layer will be considered for pruning, will be replaced
    elif compare_mode in ['local']:
        pr = base_pr[layer_index]

    # If layer name matches the pattern pre-specified in 'skip', skip it (i.e., pr = 0)
    for p in skip:
        if fnmatch(layer_name, p):
            pr = 0
    return pr

def get_pr_by_name_matching(base_pr, name, skip=[]):
    pr = 0 # Default pr = 0
    for p in base_pr:
        if fnmatch(name, p):
            pr = base_pr[p]
    for s in skip:
        if fnmatch(name, s):
            pr = 0
    return pr

def get_pr_model(layers, base_pr, skip=[], compare_mode='local'):
    r"""Get layer-wise pruning ratio for a model.
    """
    pr = OrderedDict()
    if isinstance(base_pr, str):
        ckpt = torch.load(base_pr)
        pruned, kept = ckpt['pruned_wg'], ckpt['kept_wg']
        for name in pruned:
            num_pruned, num_kept = len(pruned[name]), len(kept[name])
            pr[name] = float(num_pruned) / (num_pruned + num_kept)
        print(f"==> Load base_pr model successfully and inherit its pruning ratios: '{base_pr}'.")
    
    elif isinstance(base_pr, (float, list)):
        if compare_mode in ['global']:
            assert isinstance(base_pr, float)
            pr['model'] = base_pr
        for name, layer in layers.items():
            pr[name] = get_pr_layer(base_pr, name, layer.index, skip=skip, compare_mode=compare_mode)
        print(f"==> Get pr (pruning ratio) for pruning the model. Done. (pr may be updated later).")
    
    elif isinstance(base_pr, dict): # Index layer by name matching
        for name, layer in layers.items():
            pr[name] = get_pr_by_name_matching(base_pr, name, skip=skip)
    else:
        raise NotImplementedError
    return pr

def get_constrained_layers(layers, constrained_pattern):
    r"""Constrained layers mean those of which the pruned indices must be the same (because of the Add operator).
    """
    # 'constrained_pattern' example:
    # *layer1.0.conv3,*layer1*downsample/*layer1.2.conv3,*layer2*downsample*/*layer3.0.conv3,*layer3*downsample*/*layer4.0.conv3,*layer4*downsample*

    # Parse
    group_sep, item_sep = '/', ','
    if group_sep in constrained_pattern:
        const_groups = [x.strip() for x in constrained_pattern.split(group_sep)]
    else:
        const_groups = [constrained_pattern]
    
    # Get constrained layers
    constrained_layers = []
    for name, layer in layers.items():
        layer.constrained_group = -1
        for ix, cg in enumerate(const_groups):
            for pattern in cg.split(item_sep):
                if pattern and re.match(pattern, name):
                    layer.constrained_group = ix
                    constrained_layers += [name]

    print(f'Constrained layer pattern: [ {constrained_pattern} ]')
    return constrained_layers, layers

def adjust_pr(layers, pr, pruned, kept):
    r"""The real pr of a layer may not be exactly equal to the assigned one (i.e., raw pr) due to various reasons (e.g., constrained layers). 
    Adjust it here, e.g., averaging the prs for all constrained layers. 
    """
    pr = copy.deepcopy(pr)
    for name, layer in layers.items():
        # if name in constrained:
        #     # -- averaging within all constrained layers to keep the total num of pruned weight groups still the same
        #     num_pruned = int(num_pruned_constrained / len(constrained))
        #     # --
        #     pr[name] = num_pruned / len(layer.score)
        #     order = pruned[name] + kept[name]
        #     pruned[name], kept[name] = order[:num_pruned], order[num_pruned:]
        # else:
            num_pruned = len(pruned[name])
            pr[name] = num_pruned / len(layer.score)
    return pr

def set_same_pruned(layers, pr, pruned_wg, kept_wg, 
            constrained: [], 
            wg: str = 'filter', 
            criterion: str = 'mag', 
            sort_mode: str = 'min'):
    r"""Set pruned wgs of some layers to the same indices.
    """
    pruned_wg, kept_wg = copy.deepcopy(pruned_wg), copy.deepcopy(kept_wg)
    last_const_group = -1
    for name, layer in layers.items():
        if name in constrained:
            if layer.constrained_group != last_const_group: # A new constrained group starts
                score = get_score_layer(layer.module, wg=wg, criterion=criterion)['score']
                pruned, kept = pick_pruned_layer(score=score, pr=pr[name], sort_mode=sort_mode)
                pr_first_constrained = pr[name]
            if pr[name] != pr_first_constrained:
                print(name, layer.constrained_group, pr[name], pr_first_constrained)
            assert pr[name] == pr_first_constrained
            pruned_wg[name], kept_wg[name] = pruned, kept
            last_const_group = layer.constrained_group
    return pruned_wg, kept_wg

def get_score_layer(module, wg='filter', criterion='mag'):
    r"""Get importance score for a layer.
    
    :param: module, wg, criterion
    
    :return: A dict that has key 'score', whose value is a numpy array
    """
    # Define any scoring scheme here as you like

    #@qw: nn.MultiHeadAttention has different architecture
    if isinstance(module, nn.MultiheadAttention):
        w = module.in_proj_weight.data
    else:
        w = module.weight.data

    if wg == "channel":
        reduce_dim = [0, 2, 3] if len(w.shape) == 4 else 0
        l1 = w.abs().mean(dim=reduce_dim)
    
    elif wg == "filter":
        reduce_dim = [1, 2, 3] if len(w.shape) == 4 else 1
        l1 = w.abs().mean(dim=reduce_dim)
        if criterion == 'taylor-fo':
            g = module.accu_grad
            taylor_fo = ((w * g) ** 2).mean(dim=reduce_dim) # Eq. (8) in 2019-CVPR-Importance Estimation for Neural Network Pruning
    
    elif wg == "weight": #@qw: MHA
        if isinstance(module, nn.MultiheadAttention):
            l1 = module.in_proj_weight.abs().flatten()
        else:
            l1 = module.weight.abs().flatten()
        
        #l1 = module.weight.abs().flatten()

    out = {}
    out['mag'] = tensor2array(l1)
    out['l1-norm'] = out['mag'] # 'mag' = 'l1-norm'. Keep the 'l1-norm' key for back-compatibility
    if criterion == 'taylor-fo':
        out['taylor-fo'] = tensor2array(taylor_fo)
    out['score'] = out[criterion]
    return out

def pick_pruned_layer(score, pr=None, threshold=None, sort_mode='min', weight_shape=None):
    r"""Get the indices of pruned weight groups in a layer.

    Return: 
        pruned (list)
        kept (list)
    """
    score = np.array(score)
    num_total = len(score)
    max_pruned = int(num_total * 0.995) # This 0.995 is empirically set to avoid pruning all
    if sort_mode in ['rand'] or re.match('rand_\d+', sort_mode):
        assert pr is not None
        seed = 42
        if '_' in sort_mode:
            seed = int(sort_mode.split('_')[1])
        num_pruned = min(math.ceil(pr * num_total), max_pruned)
        np.random.seed(seed)
        order = np.random.permutation(num_total).tolist()
    
    elif sort_mode in ['min', 'max', 'ascending', 'descending']:
        num_pruned = math.ceil(pr * num_total) if threshold is None else len(np.where(score < threshold)[0])
        num_pruned = min(num_pruned, max_pruned)
        if sort_mode in ['min', 'ascending']:
            order = np.argsort(score).tolist()
        elif sort_mode in ['max', 'descending']:
            order = np.argsort(score)[::-1].tolist()
    
    elif re.match('min_\d+:\d+', sort_mode):
        # See https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
        # Currently, only unstructured pruning supports such M:N sparsity pattern
        # E.g., 'mode' = 'min_2:4'
        M, N = [int(x) for x in sort_mode.split('_')[1].split(':')]
        score = score.reshape(-1, N) # Risky, will throw an error if the total #elements is not a multiple of N 
        indices = np.argsort(score, axis=-1)[:, :M] # [M, N]
        out = []
        for row, col in enumerate(indices):
            out += (row * N + col).tolist()
        out = np.array(out)
    
    elif sort_mode in ['circu_sparsity', 'cs'] or re.match('cs_\d+', sort_mode):
        pos_1 = 0
        if '_' in sort_mode:
            pos_1 = int(sort_mode.split('_')[1])
        shape = weight_shape if len(weight_shape) == 2 else (weight_shape[0], np.prod(list(weight_shape[1:])))
        if pr > 0:
            mask = _get_sparse_circulant_matrix(shape, pr, pos_1)
            mask = mask.flatten()
            pruned = np.where(mask == 0)[0].tolist()
            kept = np.where(mask == 1)[0].tolist()
        else:
            pruned = []
            kept = list(range(len(score)))
        return pruned, kept
    else:
        raise NotImplementedError

    pruned, kept = order[:num_pruned], order[num_pruned:]
    assert isinstance(pruned, list) and isinstance(kept, list)
    return pruned, kept

def pick_pruned_model(model, layers, raw_pr, get_score_layer,
                wg: str = 'filter',
                criterion: str = 'mag',
                compare_mode: str = 'local',
                sort_mode: str = 'min',
                constrained: list = [],
                align_constrained: bool = False):
    r"""Pick pruned weight groups for a model.
    Args:
        layers: an OrderedDict, key is layer name

    Return:
        pruned (OrderedDict): key is layer name, value is the pruned indices for the layer
        kept (OrderedDict): key is layer name, value is the kept indices for the layer
    """
    assert compare_mode in ['global', 'local']
    pruned_wg, kept_wg = OrderedDict(), OrderedDict()
    all_scores, num_pruned_constrained = [], 0

    # Get importance score for each layer
    for name, module in model.named_modules():
        if name in layers:
            layer = layers[name]
            out = get_score_layer(module, wg=wg, criterion=criterion)
            if isinstance(module, nn.MultiheadAttention):
                weight_shape = module.in_proj_weight.shape
            else:
                weight_shape = module.weight.shape
            score = out['score']
            layer.score = score
            if raw_pr[name] > 0: # pr > 0 indicates we want to prune this layer so its score will be included in the <all_scores>
                all_scores = np.append(all_scores, score)

            # local pruning
            if compare_mode in ['local']:
                assert isinstance(raw_pr, dict)
                pruned_wg[name], kept_wg[name] = pick_pruned_layer(score, raw_pr[name], sort_mode=sort_mode, weight_shape=weight_shape)
                if name in constrained: 
                    num_pruned_constrained += len(pruned_wg[name])
    
    # Global pruning
    if compare_mode in ['global']:
        num_total = len(all_scores)
        num_pruned = min(math.ceil(raw_pr['model'] * num_total), num_total - 1) # do not prune all
        if sort_mode == 'min':
            threshold = sorted(all_scores)[num_pruned] # in ascending order
        elif sort_mode == 'max':
            threshold = sorted(all_scores)[::-1][num_pruned] # in decending order
        print(f'Global pruning: #all_scores {len(all_scores)}, threshold {threshold:.20f}')

        for name, layer in layers.items():
            if raw_pr[name] > 0:
                if sort_mode in ['rand']:
                    pass
                elif sort_mode in ['min', 'max']:
                    pruned_wg[name], kept_wg[name] = pick_pruned_layer(layer.score, pr=None, threshold=threshold, sort_mode=sort_mode)
            else:
                pruned_wg[name], kept_wg[name] = [], list(range(len(layer.score)))
            if name in constrained: 
                num_pruned_constrained += len(pruned_wg[name])
    
    # Adjust pr/pruned/kept
    pr = adjust_pr(layers, raw_pr, pruned_wg, kept_wg)

    if align_constrained:
        pruned_wg, kept_wg = set_same_pruned(layers, pr, pruned_wg, kept_wg, constrained, 
                                                wg=wg, criterion=criterion, sort_mode=sort_mode)

    # Print pruned indices
    if wg != 'weight':
        print(f'-' * 30 + ' Print pruned indices (start) ' + '-' * 30)
        print(f'(Note the pruned indices of the layers of the same constrained (cnst) group should be the same)')
        max_name_length = max([len(n) for n in layers])
        max_cnst_length = max([len(str(layers[n].constrained_group)) for n in layers])
        for name in layers:
            if pr[name] > 0:
                cnst_group_id = str(layers[name].constrained_group)
                print(f'{name.ljust(max_name_length)} cnst_group {cnst_group_id.rjust(max_cnst_length)} first_10_pruned_wg {pruned_wg[name][:10]}')
        print('-' * 30 + ' Print pruned indices (end) ' + '-' * 30)
    return pr, pruned_wg, kept_wg

def get_next_bn(model, layer_name):
    r"""Get the next bn layer for the layer of 'layer_name', chosen from 'model'.
    Return the bn module instead of its name.
    """
    just_passed = False
    for name, module in model.named_modules():
        if name == layer_name:
            just_passed = True
        if just_passed and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return module
    return None

def replace_module(model, name, new_m):
    r"""Replace the module <name> in <model> with <new_m>
    E.g., 'module.layer1.0.conv1' ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
    """
    obj = model
    segs = name.split(".")
    for ix in range(len(segs)):
        s = segs[ix]
        if ix == len(segs) - 1: # the last one
            if s.isdigit():
                obj.__setitem__(int(s), new_m)
            else:
                obj.__setattr__(s, new_m)
            return
        if s.isdigit():
            obj = obj.__getitem__(int(s))
        else:
            obj = obj.__getattr__(s)

def get_kept_filter_channel(layers, layer_name, pr, kept_wg, wg='filter'):
    r"""Considering layer dependency, get the kept filters and channels for the layer of 'layer_name'.
    """
    current_layer = layers[layer_name]
    if wg in ["channel"]:
        kept_chl = kept_wg[layer_name]
        next_learnable = current_layer.next
        kept_filter = list(range(current_layer.module.weight.size(0))) if next_learnable is None else kept_wg[next_learnable]
    
    elif wg in ["filter"]:
        kept_filter = kept_wg[layer_name]
        prev_learnable = current_layer.last
        if (prev_learnable is None) or pr[prev_learnable] == 0: 
            # In the case of SR networks, tail, there is an upsampling via sub-pixel. 'self.pr[prev_learnable_layer] == 0' can help avoid it. 
            # Not using this, the code will throw an error.
            cur_module = current_layer.module #@qw: include nn.MultiheadAttention
            if isinstance(cur_module, nn.MultiheadAttention):
                kept_chl = list(range(current_layer.module.in_proj_weight.size(1)))
            else:
                kept_chl = list(range(current_layer.module.weight.size(1)))
        else:
            kept_chl = kept_wg[prev_learnable]
    
    # Sort to make the indices be in ascending order 
    kept_filter, kept_chl = list(kept_filter), list(kept_chl) # Make sure they are list
    kept_filter.sort()
    kept_chl.sort()
    return kept_filter, kept_chl

def get_masks(layers, pruned_wg, wg='weight'):
    r"""Get masks for all layers in network pruning.
    """
    masks = OrderedDict()
    for name, layer in layers.items():
        pruned = pruned_wg[name]
        if wg == 'weight': # Unstructured pruning
            mask = torch.ones(layer.shape).flatten()
            mask[pruned] = 0
            mask = mask.view(layer.shape)
        elif wg == 'filter': # Structured pruning
            mask = torch.ones(layer.shape)
            mask[pruned, ...] = 0
        else:
            raise NotImplementedError
        masks[name] = mask.cuda()
    return masks

def _get_sparse_circulant_matrix(shape, sparsity_ratio, pos_1=0):
    if len(shape) == 2:
        H, W = shape
    else:
        raise NotImplementedError
    num_ = W

    # Get base sparse list for 'sparsity_ratio'
    sparsity = np.round(num_ * sparsity_ratio) / num_
    if sparsity_ratio != sparsity:
        print(f'Designated sparsity ratio rounded from {sparsity_ratio} to {sparsity}')
    sparsity_ratio = sparsity
    
    # Get the base sparse list for sparsity_ratio
    # E.g., sparsity_ratio = 0.75, num_ = 10, then first the sparsity_ratio is rounded to 0.8.
    # N0 = 8, N1 = 2, minibase = [1, 0, 0, 0, 0], base = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # E.g., sparsity_ratio = 0.625, num_ = 8, then the sparsity_ratio will still be 0.625.
    # N0 = 5, N1 = 3, minibase = [1, 0], base = [1, 0, 1, 0, 1, 0, 0, 0]
    # In each minibase, there would be only one 1 or 0
    N0, N = int(np.round(num_ * sparsity_ratio)), num_ # N0: number of zeros
    N1 = N - N0
    if N0 > N1: # Network pruning typically goes this way
        minibase = [0] + [0] * (N0 // N1)
        assert pos_1 < len(minibase)
        minibase[pos_1] = 1
        base = minibase * N1
        left = 0
    else:
        minibase = [0] + [1] * (N1 // N0)
        base = minibase * N0
        left = 1
    
    # Append the left 0 or 1's to the end
    num_left = N - len(base)
    base += [left] * num_left
    print(f'Sparsity ratio: {sparsity_ratio}, its minibase sparse list length: {len(minibase)}, pos_1: {pos_1}')
    
    # Get circulant matrix
    circ_matrix = circulant(base)
    
    # Crop or expand matrix
    if H > W:
        circ_matrix_ = circ_matrix
        for _ in range(H // W):
            circ_matrix_ = np.concatenate([circ_matrix_, circ_matrix], axis=0)
        circ_matrix = np.concatenate([circ_matrix_, circ_matrix[:H%W]], axis=0) 
    else:
        circ_matrix = circ_matrix[:H, :]
    
    assert circ_matrix.shape == (H, W)
    return circ_matrix

def set_up_prune_args(parser):
    from importlib import import_module
    from smilelogging.utils import get_arg
    from pruner import prune_method_arg

    argv = np.array(sys.argv)
    if f'--{prune_method_arg}' in argv[1:]:
        ix = np.where(argv == f'--{prune_method_arg}')[0][-1]
        method = argv[ix + 1]
        if method and not method.startswith('-'): # TODO: Add some method name check?

            # Add shared args about pruning
            from pruner.shared_args import add_args
            parser = add_args(parser)

            # Add args that are specific to the pruning method
            prune_module = import_module(f'pruner.{method}_args')
            parser = prune_module.add_args(parser)

    # Parse
    args, unknown = parser.parse_known_args()

    # Check args for pruning method
    if get_arg(args, prune_method_arg):
        from pruner.shared_args import check_args
        args = check_args(args)
        args = prune_module.check_args(args)
    return args, unknown