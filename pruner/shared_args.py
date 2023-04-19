
from pruner import prune_method_arg
import time

def add_args(parser):
    parser.add_argument(f'--{prune_method_arg}', type=str, default="",
            help='pruning method name; default is "", implying the original training without any pruning')
    parser.add_argument('--stage_pr', type=str, default="", help='to assign layer-wise pruning ratio')
    parser.add_argument('--pr_proportion', type=str, default='', help='to assign layer-wise pruning ratio proportionally')
    parser.add_argument('--compare_mode', type=str, default='local', choices=['global', 'local'])
    parser.add_argument('--index_layer', type=str, default="numbers", choices=['numbers', 'name_matching'],
            help='the rule to index layers in a network by its name; used in designating pruning ratio')
    parser.add_argument('--previous_layers', type=str, default='')
    parser.add_argument('--same_pruned_wg_layers', type=str, default='', help='constrained conv layers')
    parser.add_argument('--align_constrained', action='store_true', help='make constrained layers have the same pruned indices')
    parser.add_argument('--skip_layers', type=str, default="", help='layer id to skip when pruning')
    parser.add_argument('--reinit_layers', type=str, default="", help='layers to reinit (not inherit weights)')
    parser.add_argument('--layer_chl', type=str, default='', help='manually assign the number of channels for some layers. A not so beautiful scheme.')
    parser.add_argument('--lr_ft', type=str, default="{0:0.01,30:0.001,60:0.0001,75:0.00001}")
    parser.add_argument('--wg', type=str, default="filter", choices=['filter', 'channel', 'weight'])
    parser.add_argument('--pick_pruned', type=str, default='min', help='the criterion to select weights to prune')
    parser.add_argument('--reinit', type=str, default='', choices=['', 'default', 'pth_reset', 'xavier_uniform', 'kaiming_normal',
            'orth', 'exact_isometry_from_scratch', 'exact_isometry_based_on_existing', 'exact_isometry_based_on_existing_delta',
            'approximate_isometry', 'AI', 'approximate_isometry_from_scratch', 'AI_scratch',
            'data_dependent'],
            help='before finetuning, the pruned model will be reinited')
    parser.add_argument('--reinit_scale', type=float, default=1.)
    parser.add_argument('--base_pr_model', type=str, default='', help='the model that provides layer-wise pr')
    parser.add_argument('--inherit_pruned', type=str, default='', choices=['', 'index', 'pr'],
            help='when --base_pr_model is provided, we can choose to inherit the pruned index or only the pruning ratio (pr)')
    parser.add_argument('--model_noise_std', type=float, default=0, help='add Gaussian noise to model weights')
    parser.add_argument('--model_noise_num', type=int, default=10)
    parser.add_argument('--oracle_pruning', action="store_true")
    parser.add_argument('--ft_in_oracle_pruning', action="store_true")
    parser.add_argument('--last_n_epoch', type=int, default=5, help='in correlation analysis, collect the last_n_epoch loss and average them')
    parser.add_argument('--lr_AI', type=float, default=0.001, help="lr in approximate_isometry_optimize")
    parser.add_argument('--verbose', type=str, default='', help='print some intermediate results like gradients')
    parser.add_argument('--pre_scp', action='store_true', help='pre scp pretrained ckpt')
    parser.add_argument('--prune_schedule', type=str, default='simu', help='scheme to decide how to schedule the pruning')
    return parser

def check_args(args):
    from smilelogging.utils import parse_prune_ratio_vgg, strlist_to_list, strdict_to_dict, check_path, isfloat
    from model import num_layers, is_single_branch
    args.skip_layers = strlist_to_list(args.skip_layers, str)
    args.verbose = strlist_to_list(args.verbose, str)

    ############
    # Auto-generate constrained layers for PyTorch ResNets implementations
    # TODO: Move these to model?
    if args.same_pruned_wg_layers == 'auto':
        layers = {
            # ImageNet models
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],

            # CIFAR10 models
            'resnet56': [9, 9, 9],
            'resnet56_B': [9, 9, 9], # Same as the original resnet56 but use option-B (1x1 conv) for shortcut layers
            'resnet1202_B': [200, 200, 200],
        }
        assert args.arch in layers.keys(), "The designated '--arch' not implemented for '--same_pruned_wg_layers auto', please check"
        layers = layers[args.arch]
        last_conv = 'conv2' if args.arch in ['resnet18', 'resnet34', 'resnet56', 'resnet56_B', 'resnet1202_B'] else 'conv3'

        # Manually set, case by case
        cnst = []  # Constrained layers
        if args.arch in ['resnet56']:
            for stage_ix, num_ in enumerate(layers):
                stage_ix += 1
                if stage_ix == 1:
                    cnst += [f'module.conv1,.+layer1.\d+.{last_conv}']  # 'conv1' is also a constrained layer for CIFAR10 ResNets
                else:
                    cnst += [f'.+layer{stage_ix}.\d+.{last_conv}']

        elif args.arch in ['resnet56_B', 'resnet1202_B']:
            for stage_ix, num_ in enumerate(layers):
                stage_ix += 1
                if stage_ix == 1:
                    cnst += [f'module.conv1,.+layer1.\d+.{last_conv}'] # ResNet56_B does not have downsample layer for stage 1
                else:
                    cnst += [f'.+layer{stage_ix}.\d+.{last_conv},.+layer{stage_ix}.+downsample.+']

        else:
            for stage_ix, num_ in enumerate(layers):
                stage_ix += 1
                if stage_ix == 1:
                    cnst += [f'.+layer1.\d+.{last_conv},.+layer{stage_ix}.+downsample.+']  # 'conv1' is also a constrained layer for CIFAR10 ResNets
                else:
                    cnst += [f'.+layer{stage_ix}.\d+.{last_conv},.+layer{stage_ix}.+downsample.+']

        args.same_pruned_wg_layers = '/'.join(cnst)

    if args.same_pruned_wg_layers and not args.align_constrained:
        print(f'!! Warning: Detect --same_pruned_wg_layers used but --align_constrained not used. Is this correct?')
        time.sleep(5)
    ############

    if ':' in args.stage_pr and args.index_layer != 'name_matching':
        args.index_layer = 'name_matching'
        print(f'Warning: --stage_pr use dict-type format; the --index_layer is supposed to be `name_matching`; Has rectified this')
        # TODO-@mst: check all warnings and replace it with python logging
    try:
        args.stage_pr = check_path(args.stage_pr) # Use a ckpt to provide pr
        if args.compare_mode == 'global':
            print(f'Warning: When using a ckpt to provide pr, the "compare_mode" MUST be "local". Has corrected it to "--compare_mode local"')
            args.compare_mode = 'local'
    except ValueError:
        if isfloat(args.stage_pr):
            args.stage_pr = float(args.stage_pr) # Global pruning: only the global sparsity ratio is given
            if args.pr_proportion != '':
                args.pr_proportion = strdict_to_dict(args.pr_proportion, float)
        else:
            if args.index_layer == 'numbers': # deprecated, kept for now for back-compatability, will be removed
                if is_single_branch(args.arch): # e.g., alexnet, vgg
                    args.stage_pr = parse_prune_ratio_vgg(args.stage_pr, num_layers=num_layers[args.arch]) # example: [0-4:0.5, 5:0.6, 8-10:0.2]
                    # args.skip_layers = strlist_to_list(args.skip_layers, str) # example: [0, 2, 6]
                    if args.wg != 'weight':
                        assert args.stage_pr[num_layers[args.arch] - 1] == 0, 'The output layer should NOT be pruned. Please check your "--stage_pr" setting.'
                else: # e.g., resnet
                    args.stage_pr = strlist_to_list(args.stage_pr, float) # example: [0, 0.4, 0.5, 0]
                    # args.skip_layers = strlist_to_list(args.skip_layers, str) # example: [2.3.1, 3.1]
            elif args.index_layer == 'name_matching':
                args.stage_pr = strdict_to_dict(args.stage_pr, float)
    args.reinit_layers = strlist_to_list(args.reinit_layers, str)

    # Set up finetune lr
    assert args.lr_ft, '--lr_ft must be provided'
    args.lr_ft = strdict_to_dict(args.lr_ft, float)

    args.resume_path = check_path(args.resume_path)
    args.directly_ft_weights = check_path(args.directly_ft_weights)
    args.base_pr_model = check_path(args.base_pr_model)

    args.previous_layers = strdict_to_dict(args.previous_layers, str)

    return args
