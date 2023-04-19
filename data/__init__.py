

from importlib import import_module
import os, shutil
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader


class Data(object):
    def __init__(self, args):
        self.args = args

        # For ImageNet, the original testing takes too long time for debugging, so use a much smaller subset as proxy
        train_folder, val_folder = 'train', 'val'
        kwargs = {}
        if args.debug and args.dataset in ['imagenet', 'imagenet_subset_100', 'imagenet_subset_200', 'clip_imagenet']:
            train_folder, val_folder = 'train_tmp', 'val_tmp'
            train_folder_path = f'{args.data_path}/{args.dataset_dir}/{train_folder}'
            val_folder_path = f'{args.data_path}/{args.dataset_dir}/{val_folder}'
            # Copy train
            if not os.path.exists(train_folder_path):
                os.makedirs(train_folder_path)
                dirs = os.listdir(f'{args.data_path}/{args.dataset_dir}/train')[:3]
                [shutil.copytree(f'{args.data_path}/{args.dataset_dir}/train/{d}', f'{train_folder_path}/{d}') for d in dirs]
            # Copy val
            if not os.path.exists(val_folder_path):
                os.makedirs(val_folder_path)
                dirs = os.listdir(f'{args.data_path}/{args.dataset_dir}/val')[:3]
                [shutil.copytree(f'{args.data_path}/{args.dataset_dir}/val/{d}', f'{val_folder_path}/{d}') for d in dirs]

            kwargs = {
                'train_folder': train_folder,
                'val_folder': val_folder,
            }

        # Set up train set and test set
        dataset = import_module("data.%s" % args.dataset)
        dataset_path = os.path.join(args.data_path, args.dataset_dir)
        train_set, test_set = dataset.get_dataset(dataset_path, **kwargs)
        
        train_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        self.train_sampler = train_sampler

        self.train_loader = DataLoader(train_set,
                                       batch_size=args.batch_size,
                                       num_workers=args.workers,
                                       shuffle=(train_sampler is None),
                                       pin_memory=True,
                                       sampler=train_sampler)
        
        # The train loader for pruning algorithms, such as GReg-1/2 [Wang et al., 2021, ICLR]
        self.train_loader_prune = None


        if hasattr(args, 'batch_size_prune'):
            assert args.batch_size_prune > 0
            self.train_loader_prune = DataLoader(train_set,
                                       batch_size=args.batch_size_prune,
                                       num_workers=args.workers,
                                       shuffle=(train_sampler is None),
                                       pin_memory=True,
                                       sampler=train_sampler)
        
        self.test_loader = DataLoader(test_set,
                                      batch_size=256,
                                      num_workers=args.workers,
                                      shuffle=False,
                                      pin_memory=True)

#@qw: special settings for contrastive training 
# use CC1M for training and imagenet for testing
class conData(object):
    def __init__(self, args):
        self.args = args

        #@qw: ds1: clip_imagenet for testing
        ds1 = import_module("data.clip_imagenet")
        ds1_pth = os.path.join(args.data_path, "imagenet")
        _, test_set = ds1.get_dataset(ds1_pth)

        #@qw: ds1: conceptual caption for training
        ds2 = import_module("data.cc1m")
        ds2_pth = os.path.join(args.data_path, "cc")
        train_set = ds2.get_dataset(ds2_pth)

        train_sampler = None

        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        self.train_sampler = train_sampler
        

        self.train_loader = DataLoader(train_set,
                                       batch_size=args.batch_size,
                                       num_workers=args.workers,
                                       shuffle=(train_sampler is None),
                                       pin_memory=True,
                                       sampler=train_sampler)
        
        self.train_loader_prune = None

        if hasattr(args, 'batch_size_prune'):
            assert args.batch_size_prune > 0
            self.train_loader_prune = DataLoader(train_set,
                                       batch_size=args.batch_size_prune,
                                       num_workers=args.workers,
                                       shuffle=(train_sampler is None),
                                       pin_memory=True,
                                       sampler=train_sampler)

        
        self.test_loader = DataLoader(test_set,
                                      batch_size=256,
                                      num_workers=args.workers,
                                      shuffle=False,
                                      pin_memory=True)



#@ qw: CLIP requires specific preprocess on datasets

num_classes_dict = {
    'mnist': 10,
    'fmnist': 10,
    'kmnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet_subset_100': 100,
    'imagenet_subset_200': 200,
    'tinyimagenet': 200,
    'clip_cifar10': 10,
    'clip_cifar100': 100,
    'clip_imagenet': 1000,
    'clip_food101': 101,
    'clip_stanfordcars': 196,
    'clip_flowers102': 102,
}

# shape [N, C, H, W]
input_size_dict = {
    'mnist': (1, 1, 32, 32),
    'fmnist': (1, 1, 32, 32),
    'kmnist': (1, 1, 32, 32), 
    'cifar10': (1, 3, 32, 32),
    'cifar100': (1, 3, 32, 32),
    'imagenet': (1, 3, 224, 224),
    'imagenet_subset_100': (1, 3, 224, 224),
    'imagenet_subset_200': (1, 3, 224, 224),
    'tinyimagenet': (1, 3, 64, 64),
    'clip_cifar10': (1, 3, 224, 224),
    'clip_cifar100': (1, 3, 224, 224),
    'clip_imagenet': (1, 3, 224, 224),
    'clip_food101': (1, 3, 224, 224),
    'clip_stanfordcars': (1, 3, 224, 224),
    'clip_flowers102': (1, 3, 224, 224),
}

p1 = os.path.abspath("")
p2 = os.path.dirname(__file__)
cur_path = os.path.join(p1, p2)

prompt_path = {
    'clip_cifar10': os.path.join(cur_path, "cifar10_cls.txt"),
    'clip_cifar100': os.path.join(cur_path, "cifar100_cls.txt"),
    'clip_imagenet': os.path.join(cur_path, "ilsvrc_newcls.txt"),
    'clip_food101':os.path.join(cur_path, "food101_cls.txt"),
    'clip_stanfordcars': os.path.join(cur_path, "stanfordcars_cls.txt"),
    'clip_flowers102': os.path.join(cur_path, "flowers102_cls.txt"),
}

