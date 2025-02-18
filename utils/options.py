#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')

    # For Ensemble distillation

    parser.add_argument('--unlabeled_data_dir', type=str, default='/home/osilab7/hdd/cifar', metavar='N',
                        help='Unlabeled dataset used for ensemble')

    parser.add_argument('--unlabeled_dataset', type=str, default='cifar100', metavar='N',
                        help='Unlabeled dataset used for ensemble')

    parser.add_argument('--unlabeled_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--server_steps', type=int, default=1e4, metavar='EP',
                        help='how many steps will be trained in the server')

    parser.add_argument('--server_patience_steps', type=int, default=1e3, metavar='EP',
                        help='how many steps will be trained in the server without increase in val acc')

    parser.add_argument('--server_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate on server (default: 0.001)')

    parser.add_argument('--valid_ratio', type=float, default=0.0, metavar='LR',
                        help='Ratio of validation set')
    
    # For hetero_model
    parser.add_argument('--hetero_model', help='Model heterogenity?', default=False,
                        action='store_true')
    
    parser.add_argument('--model_group', type=int, default=0, metavar='N',
                        help='Predefined Hetero models')
    
    parser.add_argument('--local_broadcast_ep', type=int, default=5, 
                        help="the number of local epochs in broadcasting")
    
    parser.add_argument('--local_broadcast_lr', type=float, default=0.001, 
                        help="broadcasting learning rate")
    
    args = parser.parse_args()
    return args
