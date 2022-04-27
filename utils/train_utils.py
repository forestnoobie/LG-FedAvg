import json
import logging
import os
import shutil
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Custom_Nets import Custom_CNNMnist, Custom_CNNCifar

from models.Resnets import ResNet18
from models.Resnets_no_bn import resnet8_cifar
from utils.sampling import iid, noniid

import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x
    
def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    if type(y_train) == list : y_train = np.array(y_train)
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.info('Data statistics: %s' % str(net_cls_counts))
    print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

#     # Logging to console
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s',
#                                                datefmt='%Y-%m-%d %H:%M:%S'))
#     logger.addHandler(stream_handler)
        






trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_train_val(args, dataset_train):
    
    n_train = len(dataset_train)
    total_idxs = np.random.permutation(n_train)
    
    valid_n = int(args.valid_ratio * n_train)
    train_idxs = total_idxs[valid_n:]
    valid_idxs = total_idxs[:valid_n]
    
    _targets = dataset_train.targets
        
    dataset_val_ss = Subset(dataset_train, valid_idxs)
    dataset_train_ss = Subset(dataset_train, train_idxs)

    if type(_targets) == list :
        dataset_val_ss.targets = np.array(_targets)[valid_idxs]
        dataset_train_ss.targets = np.array(_targets)[train_idxs]
    else :
        dataset_val_ss.targets = _targets[valid_idxs]
        dataset_train_ss.targets = _targets[train_idxs]    
    
    return dataset_train_ss, dataset_val_ss




def get_data(args):
    
    ## Valid ratio and Subset
    
    if args.dataset == 'mnist':
        args.num_channels = 1
        dataset_train = datasets.MNIST('/home/osilab7/hdd/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('/home/osilab7/hdd/', train=False, download=True, transform=trans_mnist)
        
        dataset_val = None
        if args.valid_ratio :
            dataset_train, dataset_val = get_train_val(args, dataset_train)
        
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
            
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('/home/osilab7/hdd/cifar', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('/home/osilab7/hdd/cifar', train=False, download=True, transform=trans_cifar10_val)
        
        dataset_val = None
        if args.valid_ratio :
            dataset_train, dataset_val = get_train_val(args, dataset_train)
        
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
            
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('/home/osilab7/hdd/cifar', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('/home/osilab7/hdd/cifar', train=False, download=True, transform=trans_cifar100_val)
        
        dataset_val = None
        if args.valid_ratio :
            dataset_train, dataset_val = get_train_val(args, dataset_train)
        
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')
        
    ## Check data label distribution
    y_train = dataset_train.targets
    record_net_data_stats(y_train, dict_users_train)

    return dataset_train, dataset_test, dict_users_train, dict_users_test, dataset_val


def get_unlabeled_data(args):
    if args.unlabeled_dataset == 'mnist':
        dataset_train = datasets.MNIST('/home/osilab7/hdd/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('/home/osilab7/hdd/', train=False, download=True, transform=trans_mnist)
    
    elif args.unlabeled_dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('/home/osilab7/hdd/cifar', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('/home/osilab7/hdd/cifar', train=False, download=True, transform=trans_cifar10_val)
        
    elif args.unlabeled_dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('/home/osilab7/hdd/cifar', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('/home/osilab7/hdd/cifar', train=False, download=True, transform=trans_cifar100_val)
    
    return dataset_train, dataset_test
        

def get_model(args):
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet8' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResNet8(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet8_no_bn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = resnet8_cifar(num_classes=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob

def get_custom_model(args, model_config):
    
    if m['model_type'] == 'cnn' and args.dataset == 'mnist':
        net = CNNMnist_custom(args=args, model_config).to(args.device)
    
    return net

def get_model_hetero(args):
    hetero_models = {}
    if args.model_group == 0:
        model_group = hetero_model_group0    
    for idx, model_config in enumerate(model_group):
        hetero_model[idx] = get_custom_model(args, model_config)
    
    return hetero_models

# For 20 clients
hetero_model_group0 = dict(zip(list(range(20)) ,[{"model_type" : 'cnn', "num_layers" : 2 + i}  for i in range(4)] * 5 ))