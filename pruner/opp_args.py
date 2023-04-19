import time

def add_args(parser):
    # OPP related
    parser.add_argument('--batch_size_prune', type=int, default=256)
    parser.add_argument('--opp_scheme', type=str, default="v5", help='scheme id, used to develop new methods')
    parser.add_argument('--lw_opp', type=float, default=1000)
    parser.add_argument('--reinit_interval', type=int, default=100000000)
    parser.add_argument('--no_transpose', dest='transpose', action='store_false', default=True,
                        help='not use transpose for orth_regularization')
    parser.add_argument('--update_reg_interval', type=int, default=5, help='default for ImageNet')
    parser.add_argument('--reg_schedule', type=str, default='linear', help='linear/poly2/ploy3/...')
    parser.add_argument('--reg_iters', type=int, default=-1, help='10 epochs on ImageNet')
    parser.add_argument('--stabilize_reg_interval', type=int, default=40000)
    parser.add_argument('--lr_prune', type=float, default=0.001)
    parser.add_argument('--reg_upper_limit', type=float, default=1.0)
    parser.add_argument('--reg_granularity_prune', type=float, default=1e-4)
    parser.add_argument('--zero_out_interval', type=int, default=100, help='every zero_out_interval, zero out one layer')

    # Orthogonal regularization train
    parser.add_argument('--orth_reg_iter', type=int, default=0)
    parser.add_argument('--orth_reg_iter_ft', type=int, default=0)
    parser.add_argument('--orth_reg_method', type=str, default='CVPR20', choices=['CVPR20', 'CVPR17'])
    parser.add_argument('--lw_orth_reg', type=float, default=0.1,
            help='loss weight of orth reg. refers to CVPR20 Orthogonal-Convolutional-Neural-Networks code (14de526)')
    
    # Others for ablation studies / sanity checks
    parser.add_argument('--not_apply_reg', action="store_true", help='not apply L2 reg to gradients')
    parser.add_argument('--greg_via_loss', action="store_true", help='implement greg via loss instead of gradient')
    parser.add_argument('--no_bn_reg', dest='bn_reg', action="store_false", default=True,
        help='not apply bn reg')
    parser.add_argument('--no_weight_reg', dest='weight_reg', action="store_false", default=True,
            help='not apply weight/bias reg')
    return parser

def check_args(args):
    if args.reg_schedule in ['linear']:
        if args.reg_iters == -1:
            assert args.update_reg_interval > 0
            args.reg_iters = args.update_reg_interval * (args.reg_upper_limit / args.reg_granularity_prune)
            print('!! Warning: --reg_iters not used while --update_reg_interval used; auto-convert it to --reg_iters')
            time.sleep(5)

        if args.update_reg_interval * (args.reg_upper_limit / args.reg_granularity_prune) != args.reg_iters:
            print('!! Warning: update_reg_interval != reg_iters * reg_granularity_prune / reg_upper_limit; adjusted')
            args.update_reg_interval = args.reg_iters * args.reg_granularity_prune / args.reg_upper_limit
            time.sleep(5)

    return args
