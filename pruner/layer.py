import torch.nn as nn
import torch
import copy
from collections import OrderedDict

tensor2list = lambda x: x.data.cpu().numpy().tolist()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Layer():
    def __init__(self, name, shape, index, module, layer_type=None, last_layer=None, next_layer=None):
        self.name = name
        self.module = module
        self.shape = shape  # deprecated in support of 'shape'
        self.index = index
        self.type = layer_type
        self.last = last_layer
        self.next = next_layer


def register_modulename(model):
    for name, module in model.named_modules():
        module.name = name


def register_hooks_for_model(model, learnable_layers):
    layers = OrderedDict()
    max_len_name = [0]  # TODO-@mst: refactor this name
    max_len_shape = [0]  # Use list because this variable will be modified in hook fn
    max_len_ix = [0]
    last_module = [None]
    handles = []  # For removing hooks later

    def hook(m, i, o):
        if m.name not in layers: # To avoid replicated registration
            # Get layer parameter shape
            if hasattr(m, 'weight'):  # For CNN: naive modules like Conv2d/Linear/BN
                shape = m.weight.size()
            elif hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:  # For MHA: qkv same dim
                shape = m.in_proj_weight.size()
            elif hasattr(m, 'q_proj_weight') and m.q_proj_weight is not None:  # For MHA: qkv not same dim
                assert None not in (m.k_proj_weight, m.v_proj_weight)
                raise NotImplementedError
            else:
                raise NotImplementedError

            layer_ix = len(layers)
            shape = list(shape)
            max_len_name[0] = max(max_len_name[0], len(m.name))
            max_len_shape[0] = max(max_len_shape[0], len(str(shape)))
            max_len_ix[0] = max(max_len_ix[0], len(str(layer_ix)))
            layers[m.name] = Layer(name=m.name, shape=shape, index=layer_ix,
                                   module=m,  # TODO-@mst: Check if this takes more memory. Is this necessary?
                                   layer_type=m.__class__.__name__,
                                   last_layer=last_module[0],
            )
            last_module[0] = m.name

    def register(m, handles):
        r"""Recursively register hook for each learnable layer. A layer is defined as the node in a computation graph,
        which has no children, and has parameters.
        """
        children = list(m.children())
        if (isinstance(m, learnable_layers) and len(children) == 0) or isinstance(m, nn.MultiheadAttention):  # MHA has children
            # TODO-@mst: this MultiheadAttention manual check is ad-hoc, improve it?
            handles += [m.register_forward_hook(hook)]
            # print(f'Registered hook for layer {m.name} -- {len(layers)}') # For debug
        else:
            [register(c, handles) for c in children]

    register(model, handles)
    return layers, max_len_ix, max_len_name, max_len_shape, handles


def rm_hook(handles):
    [x.remove() for x in handles]


class LayerBuilder:
    def __init__(self, model, LEARNABLES, arch=None, dummy_input=None, input_size=None):
        register_modulename(model)
        self.model = model
        self.arch = arch
        self.LEARNABLES = LEARNABLES
        self.dummy_input = dummy_input

        self.layers, self._max_len_ix, self._max_len_name, self._max_len_shape, self.handles = \
            register_hooks_for_model(self.model, self.LEARNABLES)
        self.num_layers = len(self.layers)
        self.print_prefix = OrderedDict()
        if dummy_input is not None or input_size is not None:
            self._model_forward()
            self.finish_building_layers()

    def _model_forward(self):
        r"""Model forward to make hooks physically work.
        """
        is_train = self.model.training
        if is_train:
            self.model.eval()

        if self.dummy_input is not None:
            with torch.no_grad():
                self.model(self.dummy_input)
        elif self.input_size is not None:
            dummy_input = torch.randn(self.input_size).half().to(DEVICE)
            with torch.no_grad():
                self.model(dummy_input)

        if is_train:
            self.model.train()

    def finish_building_layers(self):
        self._max_len_ix = self._max_len_ix[0]
        self._max_len_name = self._max_len_name[0]
        self._max_len_shape = self._max_len_shape[0]
        self._adjust_layer()
        self._get_print_prefix()
        self._print_layer_stats()
        rm_hook(self.handles)

    def _get_print_prefix(self):
        for name, layer in self.layers.items():
            format_str = f"[%-{self._max_len_ix}d] %-{self._max_len_name}s %-{self._max_len_shape}s"
            self.print_prefix[name] = format_str % (layer.index, name, layer.shape)

    def _print_layer_stats(self):
        print('************************ Layer Statistics ************************')
        for name, layer in self.layers.items():
            format_str = f"%s  last: %-{self._max_len_name}s  next: %-{self._max_len_name}s"
            print(format_str % (self.print_prefix[name], layer.last, layer.next))
        print('******************************************************************')

    def _adjust_layer(self):
        r"""Add or adjust some layer information because the automatic way is not perfect now. Maybe we'll have a better
        solution.
        """
        # Add next layer
        layers_list = list(self.layers.keys()) + [None]
        for name, layer in self.layers.items():
            layer.next = layers_list[layer.index + 1]

        # Adjust last layer manually
        res50 = {
            # ResNet50, ImageNet, layers=[3,4,6,3]
            'module.layer1.0.downsample.0': 'module.conv1',
            'module.layer2.0.downsample.0': 'module.layer1.2.conv3',
            'module.layer3.0.downsample.0': 'module.layer2.3.conv3',
            'module.layer4.0.downsample.0': 'module.layer3.5.conv3',
        }
        res56_B = {
            # ResNet56_B, CIFAR10, layers=[9,9,9]
            # Downsample layer's input layer is: the last layer of the preceeding block
            'module.layer2.0.downsample.0': 'module.layer1.8.conv2',
            'module.layer3.0.downsample.0': 'module.layer2.8.conv2',
        }
        res1202_B = {
            # ResNet1202_B, CIFAR10, layers=[200,200,200]
            # Downsample layer's input layer is: the last layer of the preceeding block
            'module.layer2.0.downsample.0': 'module.layer1.199.conv2',
            'module.layer3.0.downsample.0': 'module.layer2.199.conv2',
        }
        mappings = {
            'resnet50': res50,
            'resnet56_B': res56_B,
            'resnet1202_B': res1202_B,
        }
        if self.arch in mappings:  # TODO: More general resnets?
            mp = mappings[self.arch]

            for layer_name, layer in self.layers.items():
                dp = True
                layer_name_ = layer_name
                if not layer_name.startswith('module.'):
                    dp = False
                    layer_name_ = 'module.' + layer_name

                if layer_name_ in mp:
                    prev = mp[layer_name_]
                    last = prev if dp else prev[7:]
                    layer.last = last