r"""This code is based on the official PyTorch ImageNet training example 'main.py'. Commit ID: 69d2798, 04/23/2020.
URL: https://github.com/pytorch/examples/tree/master/imagenet
Major modified parts will be indicated by '@mst' mark.
Questions to Huan Wang (wang.huan@northeastern.edu) GitHub ID: MingSun-Tse
"""

# Python packages
import os
import random
import time
import warnings
import copy, math
import numpy as np

# PyTorch packages
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as tv_models
import torch.nn.functional as F

# Our packages
from smilelogging import Logger
from smilelogging.utils import get_n_params_, get_n_flops_, Timer, get_lr
from smilelogging.utils import add_noise_to_model, _weights_init_orthogonal, get_jacobian_singular_values
from smilelogging.utils import AverageMeter, ProgressMeter, accuracy
from smilelogging.utils import EMA2 as EMA, register_ema, apply_ema, get_arg
from model import model_dict, is_single_branch
from data import Data
from data import num_classes_dict, input_size_dict, prompt_path
from pruner.reinit_model import reinit_model, rescale_model, orth_dist, deconv_orth_dist
from pruner import prune_method_arg
from option import args

# CLIP related
# from CLIP.clip import clip

# Set random seed for exact reproducing
cudnn.benchmark = True
if args.seed is not None:
    # Set seed for python libraries
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Set seed for pytorch
    torch.manual_seed(args.seed)  # Set seed for CPU
    torch.set_rng_state(torch.manual_seed(args.seed).get_state())
    torch.cuda.manual_seed(args.seed)  # Set seed for the current GPU
    torch.cuda.manual_seed_all(args.seed)  # Set seed for all the GPUs
    cudnn.benchmark = False
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

# Set up logging system (smilelogging)
pjoin = os.path.join
original_print = print
logger = Logger(args, overwrite_print=True)
netprint = logger.netprint
timer = Timer(args.epochs)

def main():
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    text_features = None

    # Init
    pruner = None
    if get_arg(args, prune_method_arg):
        prune_state = 'prune'

    # Set up data
    loader = Data(args)
    train_loader = loader.train_loader
    val_loader = loader.test_loader
    train_sampler = loader.train_sampler
    Logger.passer['train_loader'] = train_loader  # For later use
    num_classes = num_classes_dict[args.dataset]
    Logger.passer['num_classes'] = num_classes
    *_, num_channels, input_height, input_width = input_size_dict[args.dataset]

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.sd:
        kl_loss = nn.KLDivLoss(reduction="batchmean").cuda(args.gpu)
    else:
        kl_loss = None

    # TODO: Set up GPU
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Set up model (architecture)
    preset_model = False
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt)
        if args.arch not in ['clip_rn', 'clip_vit']:
            print(f'==> Load pretrained_ckpt: "{args.pretrained_ckpt}"')
            if 'model' in ckpt:
                preset_model = True
                model = ckpt['model']
                dplog = ''
                if hasattr(model, 'module'):
                    model = model.module
                    dplog = '. Found DataParallel in the model, removed'
                if hasattr(model, 'features') and hasattr(model.features, 'module'): # For back-compatibility with some old alexnet/vgg models
                    model.features = model.features.module
                    dplog = '. Found DataParallel in the model, removed'
                print(f'==> Use the model in ckpt{dplog}')
        else:
            model, text_features = load_clip(args)

    if not preset_model: # TODO-mst: Unifying this
        src = 'self-defined'
        if tv_models.__dict__.get(args.arch): # Official torchvision models
            model = tv_models.__dict__[args.arch](num_classes=num_classes, pretrained=args.pretrained)
            src = 'torchvision'
        elif args.arch in ['clip_rn', 'clip_vit']:
                model, text_features = load_clip(args)
                if args.sd: #@qw: use self distill in finetune
                    model_t, _ = load_clip(args)
                    for param in model_t.parameters():
                        param.requires_grad = False
                    print("Use self distill for finetune, teacher loaded successfully")
                else:
                    model_t = None
                    Logger.passer['text_features'] = text_features
                    Logger.passer['model_t'] = model_t
                    Logger.passer['kl_loss'] = kl_loss
        else:
            model = model_dict[args.arch](num_classes=num_classes,
                                            num_channels=num_channels,
                                            use_bn=args.use_bn,
                                            conv_type=args.conv_type)
        print(f"==> Create model '{args.arch}' (pretrained={args.pretrained}, src={src}, conv={args.conv_type})")

        if args.init in ['orth', 'exact_isometry_from_scratch']:
            model.apply(lambda m: _weights_init_orthogonal(m, act=args.activation))
            print("Use weight initialization: 'orthogonal_'. Activation: %s" % args.activation)

    # Resume weights
    if args.pretrained_ckpt:
        from collections import OrderedDict
        state_dict = OrderedDict()
        dplog = ''
        for k, v in ckpt['state_dict'].items():
            if 'module.' in k:
                k = k.replace('module.', '')
                dplog = '. Found DataParallel in the weights, removed'
            state_dict[k] = v
        model.load_state_dict(state_dict)
        print(f'==> Load pretrained weights in ckpt successfully{dplog}')
        
        #@qw: resume masks
        if 'mask' in ckpt:
            Logger.passer['mask'] = ckpt['mask']
            apply_mask_forward(model, Logger.passer['mask'])
            print("apply mask in ckpt...")

    # Set up GPU
    model = set_up_gpu(model)
    if args.sd:
        model_t = set_up_gpu(model_t)
    print('==> Set up GPU')

    # Test pretrained
    if args.test_pretrained or args.evaluate:
        if args.test_trainset: #@qw: enable evaluating model on trainset
            acc1, acc5, loss_test = validate(train_loader, model, criterion, args)
            print(f'==> Test pretrained ckpt (On train set). Acc1 {acc1:.4f} Acc5 {acc5:.4f} TestLoss {loss_test:.4f}')
        else:
            acc1, acc5, loss_test = validate(val_loader, model, criterion, args)
        print(f'==> Test pretrained ckpt. Acc1 {acc1:.4f} Acc5 {acc5:.4f} TestLoss {loss_test:.4f}')
        if args.evaluate:
            exit(0)

    # Set up optimizer
    optimizer = set_up_optimizer(model)
    print(f'==> Set up optimizer')

    start_epoch = 0
    if args.resume:
        assert args.pretrained_ckpt
        start_epoch = ckpt["epoch"]
        logstr = f'==> Resuming from epoch {ckpt["epoch"]}'

        # Resume metrics
        if 'metrics' in ckpt:
            metrics = ckpt['metrics']
            acc1 = metrics['acc1'][-1]
            logger.log_tracker.reset()
            for k, v in metrics.items():
                logger.log_tracker.update(k, v)
            logstr = f'. Metrics resumed, acc1 in ckpt = {acc1}'
            #debug
            #print(metrics['lr'])

        # Resume optimizer
        optim = ckpt['optimizer']
        optimizer.load_state_dict(optim)
        logstr += f'. Optimizer resumed'

        # Other resume options
        if get_arg(args, prune_method_arg):
            prune_state = ckpt['prune_state']
            logstr += f'. prune_state = {prune_state}'
        print(logstr)

        # # prune_state = ckpt.get('prune_state') # finetune or update_reg or stabilize_reg
        # # TODO-@mst: prune state?
        # if prune_state == 'finetune':
        #     model.load_state_dict(state['state_dict'])
        #     model = state['model'].cuda()
        #     if args.solver == 'Adam':
        #         print('==> Using Adam optimizer')
        #         optimizer = torch.optim.Adam(model.parameters(), args.lr)
        #     else:
        #         print('==> Using SGD optimizer')
        #         optimizer = torch.optim.SGD(model.parameters(), args.lr,
        #                                     momentum=args.momentum,
        #                                     weight_decay=args.weight_decay)
        #     optimizer.load_state_dict(state['optimizer'])
        #     args.start_epoch = state['epoch'] + 1
        #     print("==> Resume model successfully: '{}'. Epoch = {}. prune_state = '{}'".format(
        #             args.resume_path, args.start_epoch, prune_state))

        #     if 'mask' in state:
        #         Logger.passer['mask'] = state['mask']
        #         apply_mask_forward(model, state['mask'])
        #         print('==> Mask restored')

        # else:
        #     raise NotImplementedError

    # Set up lr scheduler
    lr_scheduler = set_up_lr_scheduler(optimizer, start_epoch)

    # Save the model after initialization
    if args.save_init_model:
        model_save = copy.deepcopy(model).cpu()
        if hasattr(model_save, 'module'):
            model_save = model_save.module
        state = {
            'arch': args.arch,
            'model': model_save,
            'state_dict': model_save.state_dict(),
            'ExpID': logger.ExpID,
        }
        save_model(state, mark='init')

    # if args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         # When using a single GPU per process and per
    #         # DistributedDataParallel, we need to divide the batch size
    #         # ourselves based on the total number of GPUs we have
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     else:
    #         model.cuda()
    #         # DistributedDataParallel will divide and allocate batch_size to all
    #         # available GPUs if device_ids are not set
    #         model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()

    if hasattr(args, 'utils') and args.utils.check_kernel_spatial_dist:
        from smilelogging.utils import check_kernel_spatial_dist
        check_kernel_spatial_dist(model)
        exit()

    module_list = nn.ModuleList([])


    # Structured pruning is basically equivalent to providing a new weight initialization before finetune,
    # so just before training, conduct pruning to obtain a new model.
    if get_arg(args, prune_method_arg):
        train_loader_prune = loader.train_loader_prune

        # Get the original unpruned model statistics
        n_params_original_v2 = get_n_params_(model)
        n_flops_original_v2 = get_n_flops_(model, img_size=input_height, n_channel=num_channels)

        # finetune a model
        if args.directly_ft_weights:
            state = torch.load(args.directly_ft_weights)
            model = torch.nn.DataParallel(ckpt['model']).cuda()
            model.load_state_dict(state['state_dict'])
            prune_state = 'finetune'
            print("==> Load model successfully: '{}'. Epoch = {}. prune_state = '{}'".format(
                args.directly_ft_weights, args.start_epoch, prune_state))

            if 'mask' in state:
                apply_mask_forward(model, state['mask'])
                print('==> Mask restored')

        if prune_state in ['prune']:
            # feature analyze
            if args.feat_analyze:
                print('analyzing feature of conv/fc .shapes (before pruning):')
                FeatureAnalyzer(model, val_loader, criterion=criterion, print=print)

            class passer:
                pass  # to pass arguments

            passer.test = validate
            passer.finetune = train
            passer.train_loader = train_loader_prune
            passer.test_loader = val_loader
            passer.save = save_model
            passer.criterion = criterion
            passer.train_sampler = train_sampler
            passer.pruner = pruner
            passer.args = args
            passer.is_single_branch = is_single_branch

            # Get dummy input
            for ix, (input, _) in enumerate(train_loader):
                dummy_input = input[0].unsqueeze(0)
                break
            if args.arch in ['clip_vit', 'clip_rn']:
                passer.dummy_input = dummy_input.half()
            else:
                passer.dummy_input = dummy_input

            # ************************* Core pruning function *************************
            from importlib import import_module
            pruner_name = get_arg(args, prune_method_arg)
            pruner_module = import_module(f'pruner.{pruner_name}_pruner')
            pruner = pruner_module.Pruner(model, loader, logger, args, passer)
            model = pruner.prune()  # Get the pruned model
            Logger.passer['pruner'] = pruner  # For later use

            if isinstance(model, tuple):
                model_before_removing_weights, model = model

            if args.wg == 'weight':
                Logger.passer['mask'] = pruner.masks
                apply_mask_forward(model, Logger.passer['mask'])
                print('==> Apply masks before finetuning to ensure the pruned weights are zero')

            netprint(model, comment='model that was just pruned')
            # *************************************************************************

            # Get model statistics of the pruned model
            n_params_now_v2 = get_n_params_(model)
            n_flops_now_v2 = get_n_flops_(model, img_size=input_height,
                                          n_channel=num_channels)  # TODO-@mst: Use torchsummaryX
            print(
                "==> n_params_original_v2: {:>9.6f}M, n_flops_original_v2: {:>9.6f}G".format(n_params_original_v2 / 1e6,
                                                                                             n_flops_original_v2 / 1e9))
            print("==> n_params_now_v2:      {:>9.6f}M, n_flops_now_v2:      {:>9.6f}G".format(n_params_now_v2 / 1e6,
                                                                                               n_flops_now_v2 / 1e9))
            ratio_param = (n_params_original_v2 - n_params_now_v2) / n_params_original_v2
            ratio_flops = (n_flops_original_v2 - n_flops_now_v2) / n_flops_original_v2
            compression_ratio = 1.0 / (1 - ratio_param)
            speedup_ratio = 1.0 / (1 - ratio_flops)
            format_str = '==> reduction ratio -- params: {:>5.2f}% (compression ratio {:>.2f}x), flops: {:>5.2f}% (' \
                         'speedup ratio {:>.2f}x) '
            print(format_str.format(ratio_param * 100, compression_ratio, ratio_flops * 100, speedup_ratio))

            # Test the just pruned model
            t1 = time.time()
            acc1, acc5, loss_test = validate(val_loader, model, criterion, args)  # test set
            logstr = "Acc1 %.4f Acc5 %.4f TestLoss %.4f" % (acc1, acc5, loss_test)
            if args.dataset not in ['imagenet'] and args.test_trainset:
                acc1_train, acc5_train, loss_train = validate(train_loader, model, criterion, args,
                                                              noisy_model_ensemble=args.model_noise_std) # train set
                logstr += " Acc1_train %.4f Acc5_train %.4f TrainLoss %.4f" % (acc1_train, acc5_train, loss_train)
            logstr += " (test_time %.2fs) Just got pruned model, about to finetune" % (time.time() - t1)
            print(logstr, acc=True)

            # Save the just pruned model
            model_save = copy.deepcopy(model).cpu()
            if hasattr(model_save, 'module'):
                model_save = model_save.module
            state = {'arch': args.arch,
                     'model': model_save,
                     'state_dict': model_save.state_dict(),
                     'acc1': acc1,
                     'acc5': acc5,
                     'ExpID': logger.ExpID,
                     'pruned_wg': pruner.pruned_wg,
                     'kept_wg': pruner.kept_wg,
                     }
            if args.wg == 'weight':
                state['mask'] = Logger.passer['mask']
            save_model(state, mark="just_finished_prune")

            if args.feat_analyze:
                if get_arg(args, prune_method_arg) in ['GReg-1', 'GReg-2']:
                    print('analyzing feature of conv/fc layers (after reg, before removing weights):')
                    FeatureAnalyzer(model_before_removing_weights, val_loader, criterion=criterion, print=print)
                print('analyzing feature of conv/fc layers (just finished pruning):')
                FeatureAnalyzer(model, val_loader, criterion=criterion, print=print)
    # ---

    # Before finetuning, we may reinit the weights by some rule
    if get_arg(args, 'reinit'):
        mask = Logger.passer['mask'] if args.wg == 'weight' else None
        model = reinit_model(model, args=args, mask=mask, print=print)
        acc1, acc5, loss_test = validate(val_loader, model, criterion, args)
        print(f"Acc1 {acc1:.4f} Acc5 {acc5:.4f} TestLoss {loss_test:.4f} -- after reiniting the just pruned model", acc=True)

        # Save weights
        model_save = copy.deepcopy(model).cpu()
        if hasattr(model_save, 'module'):
            model_save = model_save.module
        state = {'arch': args.arch,
                 'model': model_save,
                 'state_dict': model_save.state_dict(),
                 'acc1': acc1,
                 'acc5': acc5,
                 'ExpID': logger.ExpID,
                 'pruned_wg': pruner.pruned_wg,
                 'kept_wg': pruner.kept_wg,
                 }
        if args.wg == 'weight':
            state['mask'] = Logger.passer['mask']
        save_model(state, mark="reinit")
        print(f'Reinited model saved')

        if get_arg(args, 'feat_analyze'):
            print('Analyzing feature of conv/fc layers (after reinit):')
            FeatureAnalyzer(model, val_loader, criterion=criterion, print=print)

    if args.rescale:
        print(f'==> Rescale model weights, begin:')
        model = rescale_model(model, args.rescale)
        print(f'==> Rescale model weights, done!')

    if get_arg(args, prune_method_arg):
        optimizer = set_up_optimizer(model)
        print(f'==> After pruning, about to finetune, reset optimizer')

    # Check Jacobian singular value (JSV) after pruning
    if args.jsv_loop:
        if get_arg(args, prune_method_arg) in ['GReg-1', 'GReg-2']:
            jsv, jsv_diff, cn = get_jacobian_singular_values(model_before_removing_weights, train_loader,
                                                             num_classes=num_classes, n_loop=args.jsv_loop,
                                                             rand_data=args.jsv_rand_data)
            cn = [x for x in cn if not math.isnan(x)]
            print(
                'JSV_mean %.4f JSV_std %.4f JSV_std/mean %.4f JSV_max %.4f JSV_min %.4f Condition_Number_mean %.4f JSV_diff_mean %.4f JSV_diff_std %.4f -- model_before_removing_weights' %
                (np.mean(jsv), np.std(jsv), np.std(jsv) / np.mean(jsv), np.max(jsv), np.min(jsv), np.mean(cn),
                 np.mean(jsv_diff), np.std(jsv_diff)))
        jsv, jsv_diff, cn = get_jacobian_singular_values(model, train_loader, num_classes=num_classes,
                                                         n_loop=args.jsv_loop, rand_data=args.jsv_rand_data)
        cn = [x for x in cn if not math.isnan(x)]
        print(
            'JSV_mean %.4f JSV_std %.4f JSV_std/mean %.4f JSV_max %.4f JSV_min %.4f Condition_Number_mean %.4f JSV_diff_mean %.4f JSV_diff_std %.4f' %
            (np.mean(jsv), np.std(jsv), np.std(jsv) / np.mean(jsv), np.max(jsv), np.min(jsv), np.mean(cn),
             np.mean(jsv_diff), np.std(jsv_diff)))

        # For easy report
        Logger.passer['JSV_mean'] = [np.mean(jsv)]
        Logger.passer['JSV_std/mean'] = [np.std(jsv) / np.mean(jsv)]

    # EMA
    if args.ema > 0:
        ema_set = [[model, EMA(args.ema)]]
        Logger.passer['ema_set'] = ema_set
        register_ema(ema_set)

    # Track the grad trajectory
    if hasattr(args, 'utils') and args.utils.check_grad_history > 0:
        grad_history = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                grad_history[name] = []
        keys = list(grad_history.keys())
        if len(keys) > 10:  # At most, track 10 layers
            interval = len(keys) // 10
            keys = keys[::interval]
            grad_history = {k: v for k, v in grad_history.items() if k in keys}
        Logger.passer['grad_history'] = grad_history

    # Main train fn
    train(model, optimizer, lr_scheduler, train_loader, val_loader, train_sampler, criterion, pruner, start_epoch)

def set_up_optimizer(model):
    lr = args.lr_ft if hasattr(args, 'lr_ft') else args.lr
    init_lr = list(lr.values())[0]
    if args.solver == 'Adam':
        print('Use Adam optimizer')
        optim = torch.optim.Adam(model.parameters(), init_lr)
    else:
        print('Use SGD optimizer')
        optim = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optim


def set_up_lr_scheduler(optimizer, start_epoch):
    if hasattr(args, 'advanced_lr') and args.advanced_lr.lr_decay == 'cos_v2':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        min_lr = args.advanced_lr.min_lr
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=min_lr, last_epoch=start_epoch - 1)
        print(
            f'==> Create lr scheduler: CosineAnnealingLR, eta_min={min_lr}, T_max: {args.epochs}, last_epoch: {start_epoch - 1}')
    else:
        from smilelogging.utils import PresetLRScheduler
        lr = args.lr_ft if get_arg(args, 'lr_ft') else args.lr
        lr_scheduler = PresetLRScheduler(lr)
        print(
            f'==> Create lr scheduler: Step LR, {lr}')  # TODO-@mst: Mimic pytorch lr scheduler, implemement a new one; use --lr_schedule
    return lr_scheduler


def train(model, optimizer, lr_scheduler, train_loader, val_loader, train_sampler, criterion, pruner, start_epoch):
    metrics = logger.log_tracker.get_metrics()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # LR scheduling
        if not hasattr(args, 'advanced_lr'):
            lr_scheduler(optimizer, epoch)
        else:
            if args.advanced_lr.lr_decay == 'cos_v2':
                lr_scheduler.step()
        # Get LR
        lr = get_lr(optimizer)

        # Save model if LR just changed; currently only for Step LR scheduler
        #debug
        #print(metrics['lr'])

        #@qw: the desired metrics['lr'] should be in the shape like [0.001, 0.001, 0.001], while sometims metrics['lr'] could be in the shape like [[0.001, 0.001, 0.001]]
        if not hasattr(args, 'advanced_lr'):
            if metrics.get('lr') is None:
                last_lr = -1
            else:
                try:
                    last_lr =  metrics['lr'][0][-1] 
                except:
                    last_lr =  metrics['lr'][-1] 

            if last_lr != -1 and lr != last_lr:
                save_model(ckpt, mark=f'lr{last_lr}_epoch{epoch}')
                print(f'==> Save ckpt at the last epoch ({epoch}) of LR {last_lr}')

        # One epoch train
        logger.log_tracker.update('lr', lr)
        print("==> Set lr = %s @ Epoch %d (Start)" % (lr, epoch + 1))
        one_epoch_train(train_loader, model, criterion, optimizer, epoch)

        # Check weights magnitude during finetune
        if args.__dict__.get(prune_method_arg) in ['GReg-1', 'GReg-2'] and not isinstance(pruner, type(None)):
            for name, m in model.named_modules():
                if name in pruner.reg:
                    ix = pruner.layers[name].index
                    mag_now = m.weight.data.abs().mean()
                    mag_old = pruner.original_w_mag[name]
                    ratio = mag_now / mag_old
                    tmp = '[%2d] %25s -- mag_old = %.4f, mag_now = %.4f (%.2f)' % (ix, name, mag_old, mag_now, ratio)
                    print(tmp, unprefix=True)

        # Test
        acc1, acc5, loss_test = validate(val_loader, model, criterion, args)
        if args.dataset not in ['imagenet'] and args.test_trainset:  # Too costly, not test for now
            acc1_train, acc5_train, loss_train = validate(train_loader, model, criterion, args)
        else:
            acc1_train, acc5_train, loss_train = -1, -1, -1

        # Log down metrics
        logger.log_tracker.update('acc1', acc1)
        logger.log_tracker.update('acc5', acc5)
        logger.log_tracker.update('loss_test', loss_test)

        # Print metrics
        metrics = logger.log_tracker.get_metrics()
        is_best = len(metrics['acc1']) == 1 or acc1 > metrics['acc1'][:-1].max()
        best_acc1 = metrics['acc1'].max()
        best_acc1_epoch = metrics['acc1'].tolist().index(best_acc1)
        acclog = "Epoch %d Acc1 %.4f Acc5 %.4f TestLoss %.4f BestAcc1 %.4f @ BestAcc1Epoch %d LR %s" % \
                 (epoch + 1, acc1, acc5, loss_test, best_acc1, best_acc1_epoch + 1, lr)
        if acc1_train != -1:
            acclog.replace('BestAcc1', f'TrainAcc1 {acc1_train:.4f} TrainAcc5 {acc5_train:.4f} TrainLoss {loss_train:.4f} BestAcc1')

        print(acclog, acc=True)
        print(f'Predicted finish time: {timer()}')

        # Save
        ngpus_per_node = torch.cuda.device_count()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):

            model_save = copy.deepcopy(model).cpu()
            if hasattr(model_save, 'module'):
                model_save = model_save.module # Remove data parallel, which usually is a trouble when loading weights
            ckpt = {'epoch': epoch + 1,
                    'arch': args.arch,
                    'model': model_save,
                    'state_dict': model_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ExpID': logger.ExpID,
                    'prune_state': 'finetune',
                    'lr': lr,
                    'metrics': metrics,
                    }
            if get_arg(args, prune_method_arg) and args.wg == 'weight':
                ckpt['mask'] = Logger.passer['mask']
            mark = f'epoch{epoch + 1}' if args.save_model_interval > 0 and (epoch + 1) % args.save_model_interval == 0 else ''
            save_model(ckpt, is_best, mark=mark)

def one_epoch_train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model_t, kl_loss = None, None
    if args.arch in ['clip_rn', 'clip_vit']:
        from smilelogging import Logger
        text_features = Logger.passer['text_features']
        model_t = Logger.passer['model_t']
        kl_loss = Logger.passer['kl_loss']

    # Switch to train mode
    model.train()
    if model_t:
        model_t.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        i = i + 1

        # measure data loading time
        data_time.update(time.time() - end)

        images, target = images.cuda(), target.cuda()
        if args.half_input:
            images = images.half()

        if hasattr(args, 'advanced_lr') and args.advanced_lr.lr_decay != 'cos_v2':
            lr = adjust_learning_rate_v2(optimizer, epoch, i, len(train_loader))
            args.advanced_lr.lr = lr
            if i == 10:
                print(f'==> Set LR to {lr:.6f} Epoch {epoch} Iter {i}')

        # compute output
        # @qw: special forward for CLIP
        if args.arch in ['clip_rn', 'clip_vit']:
            output = clip_fw(model, text_features, images)
        else:
            output = model(images)
        loss = criterion(output, target)

        if model_t and kl_loss: #self distill
            #print("Engage self distill")
            model_t_out = clip_fw(model_t, text_features, images)
            kldiv = kl_loss(F.log_softmax(output / args.temperature, dim=1), F.softmax(model_t_out / args.temperature, dim=1)) * (T ** 2)
            loss = (1 - args.alpha) * loss + args.alpha * kldiv

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Orthogonal regularization
        if get_arg(args, 'orth_reg_iter_ft'):
            loss_orth_reg, cnt = 0, -1
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    cnt += 1
                    if args.orth_reg_method in ['CVPR20']:
                        if cnt != 0:  # per the CVPR20 paper, do not reg the 1st conv
                            shape = module.weight.shape
                            if len(shape) == 2 or shape[-1] == 1:  # FC and 1x1 conv
                                loss_orth_reg += orth_dist(module.weight)
                            else:
                                loss_orth_reg += deconv_orth_dist(module.weight)
                    elif args.orth_reg_method in ['CVPR17']:
                        loss_orth_reg += orth_dist(module.weight)
                    else:
                        raise NotImplementedError
            loss += loss_orth_reg * args.lw_orth_reg
            if i % args.print_interval == 0:
                print(f'loss_orth_reg (*{args.lw_orth_reg}) {loss_orth_reg:.10f} Epoch {epoch} Iter {i}')

        # Collect weights before update
        if hasattr(args, 'utils') and args.utils.check_grad_history:
            params_before = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    params_before[name] = module.weight.data.clone()

        # Compute gradient and update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # After update, zero out pruned weights
        if get_arg(args, prune_method_arg) and args.wg == 'weight':
            from smilelogging import Logger
            masks = Logger.passer['mask']
            apply_mask_forward(model, masks)

        # -- There seems to be bugs in this impl., not used for now
        # # Collect weights after update
        # if hasattr(args, 'utils') and args.utils.check_grad_history:
        #     effective_grad = {}
        #     from smilelogging.utils import get_lr
        #     lr = get_lr(optimizer)
        #     for name, module in model.named_modules():
        #         if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        #             effective_grad[name] = (params_before[name] - module.weight.data.clone()) / lr
        #             # if i % 100 == 0 and '.10' in name:
        #             #     print('effective_grad', effective_grad[name])
        # --

        # Apply EMA
        if args.ema > 0:
            ema_set = Logger.passer['ema_set']
            apply_ema(ema_set)

        # Utils: check gradient norm
        if hasattr(args, 'utils') and args.utils.check_grad_norm:
            from smilelogging.utils import check_grad_norm
            if i % args.print_interval == 0:
                print('');
                print(f'(** Start check_grad_norm. Epoch {epoch} Step {i} **)')
                check_grad_norm(model)
                print(f'(** End check_grad_norm **)');
                print('')

        # Utils: check gradient stats
        if hasattr(args, 'utils') and args.utils.check_grad_stats:
            from smilelogging.utils import check_grad_stats
            if i % args.print_interval == 0:
                print('');
                print(f'(** Start check_grad_stats. Epoch {epoch} Step {i} **)')
                check_grad_stats(model)
                print(f'(** End check_grad_stats **)');
                print('')

        # Utils: Check grad history
        if hasattr(args, 'utils') and args.utils.check_grad_history:
            grad_history = Logger.passer['grad_history']
            pruner = Logger.passer['pruner']
            assert args.wg == 'weight'
            for name, module in model.named_modules():
                if name in grad_history:
                    grad = module.weight.grad.data.clone().cpu()
                    # grad = effective_grad[name].clone().cpu()
                    grad_history[name] += [grad]

            # Print
            window_size, ep = 100, 1e-30
            if i > 0 and i % window_size == 0:
                print('');
                print(f'(** Start check_grad_history. Epoch {epoch} Step {i} **)')
                for name, module in model.named_modules():
                    if name in grad_history:
                        # Get SNR
                        grad_history_ = torch.stack(grad_history[name], dim=0)
                        grad_std = grad_history_.std(dim=0)  # Std along the example axis
                        grad_abs_mean = grad_history_.abs().mean(dim=0)  # Mean along the example axis
                        snr = grad_abs_mean / (grad_std + ep)
                        # print(f'grad_std {grad_std}')

                        # Only consider the unmasked grads
                        mask = Logger.passer['mask'][name].flatten()
                        snr_ = [snr.flatten()[i].item() for i in range(mask.numel()) if mask[i]]
                        snr = torch.Tensor(snr_).mean()

                        print(
                            f'[{name:>20s} Epoch {epoch} Step {i}] avg_grad_abs_mean {grad_abs_mean.mean():.8f} avg_grad_std {grad_std.mean():.8f} avg_grad_snr {snr:.8f} #snr {len(snr_)} pr {pruner.pr[name]} #params {mask.numel()}')
                        grad_history[name] = []  # Clear
                print(f'(** End check_grad_history **)');
                print('')

        # Utils: check weight stats
        if hasattr(args, 'utils') and args.utils.check_weight_stats:
            from smilelogging.utils import check_weight_stats
            if i % args.print_interval == 0:
                print('');
                print(f'(** Start check_weight_stats. Epoch {epoch} Step {i} **)')
                check_weight_stats(model)
                print(f'(** End check_weight_stats **)');
                print('')

        # @mst: check Jacobian singular value (JSV)
        if args.jsv_interval == -1:
            args.jsv_interval = len(train_loader)  # default: check jsv at the last iteration
        if args.jsv_loop and (i + 1) % args.jsv_interval == 0:
            from smilelogging import Logger
            jsv, jsv_diff, cn = get_jacobian_singular_values(model, train_loader,
                                                             num_classes=Logger.passer['num_classes'],
                                                             n_loop=args.jsv_loop,
                                                             rand_data=args.jsv_rand_data)
            print('JSV_mean %.4f JSV_std %.4f JSV_std/mean %.4f JSV_max %.4f JSV_min %.4f Condition_Number_mean %.4f \
JSV_diff_mean %.4f JSV_diff_std %.4f -- Epoch %d Iter %d' %
                  (np.mean(jsv), np.std(jsv), np.std(jsv) / np.mean(jsv), np.max(jsv), np.min(jsv), np.mean(cn),
                   np.mean(jsv_diff), np.std(jsv_diff), epoch, i))

            # For easy report
            Logger.passer['JSV_mean'] += [np.mean(jsv)]
            Logger.passer['JSV_std/mean'] += [np.std(jsv) / np.mean(jsv)]
            if len(Logger.passer['JSV_mean']) == 11:
                logstr = []
                for x, y in zip(Logger.passer['JSV_mean'], Logger.passer['JSV_std/mean']):
                    logstr += ['%.4f/%.2f' % (x, y)]
                logstr = ' | '.join(logstr) + ' |'  # For markdown
                print(f'First 10-epoch JSV_mean, JSV_std/mean: {logstr}')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, noisy_model_ensemble=False, print=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    train_state = model.training

    # switch to evaluate mode
    model.eval()

    # @mst: add noise to model
    model_ensemble = []
    if noisy_model_ensemble:
        for i in range(args.model_noise_num):
            noisy_model = add_noise_to_model(model, std=args.model_noise_std)
            model_ensemble.append(noisy_model)
        print(
            '==> added Gaussian noise to model weights (std=%s, num=%d)' % (args.model_noise_std, args.model_noise_num))
    else:
        model_ensemble.append(model)

    # Pass variables
    from smilelogging import Logger
    if 'text_features' in Logger.passer:
        text_features = Logger.passer['text_features']

    time_compute = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            images, target = images.cuda(), target.cuda()
            if args.half_input:
                images = images.half()

            # compute output
            t1 = time.time()
            output = 0
            for model in model_ensemble:  # @mst: test model ensemble
                if args.arch in ['clip_rn', 'clip_vit']:
                    output += clip_fw(model, text_features, images)
                else:
                    output += model(images)
            output /= len(model_ensemble)
            time_compute.append((time.time() - t1) / images.size(0))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if print and i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        # @mst: commented because we will use another print outside 'validate'
    # print("time compute: %.4f ms" % (np.mean(time_compute)*1000))

    # change back to original model state if necessary
    if train_state:
        model.train()
    return top1.avg.item(), top5.avg.item(), losses.avg  # @mst: added returning top5 acc and loss


def save_model(ckpt, is_best=False, mark=''):
    out = pjoin(logger.weights_path, "ckpt.pth")
    torch.save(ckpt, out)
    if is_best:
        out_best = pjoin(logger.weights_path, "ckpt_best.pth")
        torch.save(ckpt, out_best)
    if mark:
        out_mark = pjoin(logger.weights_path, "ckpt_{}.pth".format(mark))
        torch.save(ckpt, out_mark)


# Zero out pruned weights for unstructured pruning
def apply_mask_forward(model, masks):
    for name, m in model.named_modules():
        if name in masks:
            if isinstance(m, nn.MultiheadAttention):
                m.in_proj_weight.data.mul_(masks[name])
            else:
                m.weight.data.mul_(masks[name])


def adjust_learning_rate_v2(optimizer, epoch, iteration, num_iter):
    r"""More advanced LR scheduling. Refers to d-li14 MobileNetV2 ImageNet implementation:
    https://github.com/d-li14/mobilenetv2.pytorch/blob/1733532bd43743442077326e1efc556d7cfd025d/imagenet.py#L374
    """
    assert hasattr(args, 'advanced_lr')

    warmup_iter = args.advanced_lr.warmup_epoch * num_iter  # num_iter: num_iter_per_epoch
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if epoch < args.advanced_lr.warmup_epoch:
        lr = args.lr * current_iter / warmup_iter
    else:
        if args.advanced_lr.lr_decay == 'step':
            lr = args.lr * (args.advanced_lr.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
        elif args.advanced_lr.lr_decay == 'cos':
            lr = args.lr * (1 + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        elif args.advanced_lr.lr_decay == 'linear':
            lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
        elif args.advanced_lr.lr_decay == 'schedule':
            count = sum([1 for s in args.advanced_lr.schedule if s <= epoch])
            lr = args.lr * pow(args.advanced_lr.gamma, count)
        else:
            raise ValueError('Unknown lr mode {}'.format(args.advanced_lr.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_up_gpu(model):
    model = nn.DataParallel(model)
    model.cuda()
    return model

def load_clip(arg):
    # load pretrained CLIP model
    if arg.arch == 'clip_vit':
        net, _ = clip.load("ViT-B/16")
    if arg.arch == 'clip_rn':
        net, _ = clip.load("RN101")

    #vision
    vision_model = model_dict[arg.arch](net)

    #language
    f = open(prompt_path[arg.dataset],'r')
    classes = [s.strip() for s in f.readlines()]
    if arg.dataset == 'clip_stanfordcars':
        text_descriptions = [f"i love my {label}!" for label in classes]
    else:
        text_descriptions = [f"itap of a {label}!" for label in classes] #itap of a; i love my
    text_tokens = clip.tokenize(text_descriptions).cuda()
    with torch.no_grad():
        text_features = net.encode_text(text_tokens).half()
        text_features /= text_features.norm(dim = -1, keepdim = True)
    print(text_features.shape)
    
    return vision_model, text_features

def clip_fw(vnet, text, input): # vision model, text_features, batched input
    vision_out = vnet(input)
    res = 100.0 * vision_out @ text.T
    return res




if __name__ == '__main__':
    # Check data
    data_script = 'scripts/set_up_data.sh'
    if os.path.exists(data_script):
        os.system(f'sh {data_script} {args.dataset}')

    # Scp experiment
    scp_script = 'scripts/scp_experiments_to_hub.sh'
    if os.path.exists(scp_script):
        from smilelogging.utils import scp_experiment
        need_scp = scp_experiment(scp_script, logger, args)
        if need_scp:
            print('==> Initial scp done')

    main()

    # Scp experiment
    scp_script = 'scripts/scp_experiments_to_hub.sh'
    if os.path.exists(scp_script):
        from smilelogging.utils import scp_experiment
        need_scp = scp_experiment(scp_script, logger, args, mv=True)
        if need_scp:
            print('==> Final scp done')
