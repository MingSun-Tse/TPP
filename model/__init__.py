from importlib import import_module
from .vgg import vgg11, vgg13, vgg16, vgg19
from .resnet_cifar10 import resnet20, resnet56, resnet56_B, resnet56x4, resnet110_B, resnet1202_B
from .resnet_cifar10_Jonathan import resnet20 as resnet20_J, resnet56 as resnet56_J
from .lenet5 import lenet5, lenet5_mini, lenet5_linear, lenet5_wider_linear, lenet5_wider_linear_nomaxpool
from .mlp import mlp_7_linear, mlp_7_relu, mlp_50_linear, mlp_20_linear
from .clip_img import ImageCLIP, TextCLIP

def set_up_model(args, logger):
    logger.log_printer("==> making model ...")
    module = import_module("model.model_%s" % args.method)
    model = module.make_model(args, logger)    
    return model

def is_single_branch(model_name):
    for k in single_branch_model:
        if model_name.startswith(k):
            return True
    return False


model_dict = {
    # MNIST models
    'mlp_7_linear': mlp_7_linear,
    'mlp_50_linear': mlp_50_linear,
    'mlp_20_linear': mlp_20_linear,
    'mlp_7_relu': mlp_7_relu,
    'lenet5': lenet5,
    'lenet5_mini': lenet5_mini,
    'lenet5_linear': lenet5_linear,
    'lenet5_wider_linear': lenet5_wider_linear,
    'lenet5_wider_linear_nomaxpool': lenet5_wider_linear_nomaxpool,

    # CIFAR resnet models
    'resnet56': resnet56,
    'resnet56_B': resnet56_B,
    'resnet56x4': resnet56x4,
    'resnet110_B': resnet110_B,
    'resnet1202_B': resnet1202_B,
    'resnet20_J': resnet20_J,

    # CIFAR vgg models
    'vgg11_C': vgg11,
    'vgg13_C': vgg13,
    'vgg16_C': vgg16,
    'vgg19_C': vgg19,

    'clip_rn': ImageCLIP,
    'clip_vit': ImageCLIP,
    'text_enc': TextCLIP
}

num_layers = {
    'mlp_7_linear': 7,
    'mlp_50_linear': 50,
    'mlp_20_linear': 20,
    'mlp_7_relu': 7,
    'lenet5': 5,
    'lenet5_mini': 5,
    'lenet5_linear': 5,
    'lenet5_wider_linear': 5,
    'lenet5_wider_linear_nomaxpool': 5,
    'alexnet': 8,

    # -------------------
    # These VGG nets are from [EigenDamage, ICML, 2019], where the two FC layers are replaced with a global average pooling,
    # thus they have two fewer layers than the original VGG nets.
    'vgg11_C': 9,
    'vgg13_C': 11,
    'vgg16_C': 14,
    'vgg19_C': 17,
    # -------------------

    'vgg11_bn': 11,
    'vgg13_bn': 13,
    'vgg16_bn': 16,
    'vgg19_bn': 19,
}

single_branch_model = [
    'mlp',
    'lenet',
    'alexnet',
    'vgg',
]