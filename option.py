import torchvision.models as models
from smilelogging.utils import strdict_to_dict

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def add_args(parser):
    # Model related args
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        # choices=model_names, # @mst: We will use more than the imagenet models, so remove this
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--conv_type', type=str, default='default', choices=['default', 'wn'])
    parser.add_argument('--not_use_bn', dest='use_bn', default=True, action="store_false", help='if use BN in the network')
    parser.add_argument('--activation', type=str, default='relu', help="activation function",
                        choices=['relu', 'leaky_relu', 'linear', 'tanh', 'sigmoid'])

    # Data related args
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--dataset',
                        help='dataset name', default='imagenet')
    parser.add_argument('--dataset_dir',
                        help='path of dataset folder', default=None)


    # Training related args
    parser.add_argument('--init', type=str, default='default', help="parameter initialization scheme")
    parser.add_argument('--lr', type=str, default='0:0.1', metavar='LR')
    parser.add_argument('-b', '--batch-size', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                       help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', type=float, default=0.0001)
    parser.add_argument('--solver', '--optim', type=str, default='SGD')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=2000)
    parser.add_argument('--plot_interval', type=int, default=100000000)
    parser.add_argument('--save_model_interval', type=int, default=-1, help="the interval to save model")
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--resume_path', type=str, default='', help="supposed to replace the original 'resume' feature")
    parser.add_argument('--pretrained_ckpt', type=str, default='', help="supposed to replace the original 'resume' feature")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--directly_ft_weights', type=str, default='', help="the path to a pretrained model")
    parser.add_argument('--test_pretrained', action="store_true", help='test the pretrained model')
    parser.add_argument('--save_init_model', action="store_true", help='save the model after initialization')

    #Self distill related args
    #@qw: attempt to adopt self distill for finetuning to improve zero-shot classification accuracy (designed for CLIP)
    parser.add_argument('--sd', action="store_true", help='adopt self distill for finetuning')
    parser.add_argument('--alpha', default=0.5, type=float, help='distill loss weight')
    parser.add_argument('--temperature', default=4, type=int, help='distill temperature')

    #@qw: contrastive training?
    parser.add_argument('--ctv', action="store_true", help='adopt contrastive training')

    # GPU/Parallel related args
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--half_input', action='store_true',
                        help='GPU id to use.')

    # Advanced LR scheduling related
    parser.add_argument('--advanced_lr.ON', action="store_true")
    parser.add_argument('--advanced_lr.lr_decay', type=str, choices=['step', 'cos', 'cos_v2', 'linear', 'schedule'])
    parser.add_argument('--advanced_lr.warmup_epoch', type=int, default=0)
    parser.add_argument('--advanced_lr.min_lr', type=float, default=1e-5)

    # This code base also serves to quick-check properties of deep neural networks. These functionalities are summarized here.
    parser.add_argument('--utils.ON', action="store_true")
    parser.add_argument('--utils.check_kernel_spatial_dist', action="store_true")
    parser.add_argument('--utils.check_grad_norm', action="store_true")
    parser.add_argument('--utils.check_grad_stats', action="store_true")
    parser.add_argument('--utils.check_grad_history', action="store_true")
    parser.add_argument('--utils.check_weight_stats', action="store_true")

    # Other args for analysis
    parser.add_argument('--rescale', type=str, default='')
    parser.add_argument('--jsv_loop', type=int, default=0, help="num of batch loops when checking Jacobian singuar values")
    parser.add_argument('--jsv_interval', type=int, default=-1, help="the interval of printing jsv")
    parser.add_argument('--jsv_rand_data', action="store_true", help='if use data in random order to check JSV')
    parser.add_argument('--feat_analyze', action="store_true", help='analyze features of conv/fc layers')
    parser.add_argument('--test_trainset', action="store_true")
    parser.add_argument('--ema', type=float, default=0)
    return parser

def check_args(args):
    from smilelogging.utils import update_args, check_path
    import os, glob
    args.lr = strdict_to_dict(args.lr, float)

    # ======
    # Check pretrained ckpt; fetch it if it is unavailable locally
    if args.pretrained_ckpt:
        print(f'==> Checking pretrained_ckpt at path "{args.pretrained_ckpt}"', end='', flush=True)
        candidates = glob.glob(args.pretrained_ckpt)
        if len(candidates) == 0:
            print(', not found it. Fetching it...', end='', flush=True)
            keyword = args.pretrained_ckpt.split(os.sep)[1]
            script = f'sh scripts/scp_experiments_from_hub.sh {args.experiments_dir} {keyword}'
            os.system(script)
            print(', fetch it done!', flush=True)
        elif len(candidates) == 1:
            print(', found it!', flush=True)
        elif len(candidates) > 1:
            print(', found more than 1 ckpt candidates; please check --pretrained_ckpt', flush=True)
            exit(1)
        args.pretrained_ckpt = check_path(args.pretrained_ckpt)
    # ======

    if args.dataset_dir is None:
        args.dataset_dir = args.dataset

    if 'linear' in args.arch.lower():
        args.activation = 'linear'

    # Check arch name
    if args.dataset in ['cifar10', 'cifar100'] and args.arch.startswith('vgg') and not args.arch.endswith('_C'):
        print(f'==> Error: Detected CIFAR dataset used while the VGG net names do not end with "_C". Fix this, e.g., '
              f'change vgg19 to vgg19_C')
        exit(1)

    # Some deprecated args to maintain back-compatibility
    args.copy_bn_w = True
    args.copy_bn_b = True
    args.reg_multiplier = 1

    # Update args to enable some advanced features
    args = update_args(args)
    return args

def check_unknown(unknown, debug):
    if len(unknown):
        print(f'Unknown args. Please check in case of unexpected setups: {unknown}')
        
        # Check unknown args in case of wrong setups
        # TODO-@mst: this is a bit ad-hoc, a better solution?
        if '--base_model_path' in unknown:
            print(f'Error: "--base_model_path" is retired, use "--pretrained_ckpt" instead')
        if '--wd' in unknown:
            print(f'Error: "--wd" is retired, use "--weight_decay" instead')
    
        if not debug:
            exit(1)

from smilelogging import argparser as parser
parser = add_args(parser) # These args are those independent of pruning

# -- This part is the key to set up pruning related args.
# If no pruning method is used, their args will not be added to parser.
from pruner.prune_utils import set_up_prune_args
args, unknown = set_up_prune_args(parser)
# --

check_unknown(unknown, args.debug)
args = check_args(args)
