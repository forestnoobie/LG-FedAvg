#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import random
import logging
import numpy as np
import pandas as pd
import torch
import numpy as np

from utils.options import args_parser
from utils.train_utils import get_data, get_unlabeled_data, get_model, set_logger
from models.Update import LocalUpdate, GlobalUpdateEns
from models.test import test_img
import os

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}_shard{}_val{}_{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.valid_ratio, args.results_save)
    fed_name = 'fedhetero'
    if not os.path.exists(os.path.join(base_dir, fed_name)):
        os.makedirs(os.path.join(base_dir, fed_name), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test, dataset_val = get_data(args) # For val data
    dataset_unlabeled_train, _ = get_unlabeled_data(args)
    
    dict_save_path = os.path.join(base_dir, '{}/dict_users.pkl'.format(fed_name))
    log_save_path = os.path.join(base_dir, '{}/log.log'.format(fed_name))
    set_logger(log_save_path)
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    logging.info(args)
    logging.info(dict_users_train)
    
    # build model
    net_glob = get_model(args)
    net_glob.train()

    # build hetero clients_
    dict_users_model = get_hetero_model(args) '''hetero'''
    
    # training
    results_save_path = os.path.join(base_dir, '{}/results.csv'.format(fed_name))

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    for iter in range(args.epochs):
        w_glob = None
        selected_locals = []
        
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        np.random.seed(iter)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        logging.info("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            
            '''BroadCast'''
            broad_hetero = BroadcastHetero(args=args, dataset=dataset_train, idxs=dict_users_train[idx]) # To Do
            w_glob, epoch_loss = broad_hetero.train(net_local=dict_users_model[idx], net_glob=net_glob)
            #net_local = copy.deepcopy(net_glob) 
            
            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss)) 
            selected_locals.append(copy.deepcopy(w_local))

            # if w_glob is None:
            #     w_glob = copy.deepcopy(w_local)
            # else:
            #     for k in w_glob.keys():
            #         w_glob[k] += w_local[k] 

        lr *= args.lr_decay

        '''Aggregation'''
        # update global weights
        # for k in w_glob.keys():
        #     w_glob[k] = torch.div(w_glob[k], m)
        global_update = GlobalUpdateEnsHetero(args=args, dataset=dataset_unlabeled_train, val_data=dataset_val,
                              idx_users=idx_users, dict_users_model=dict_users_model)
        global_update.train(net_glob=net_glob, args=args)

        # copy weight to net_glob
        #net_glob.load_state_dict(w_glob) ## Aggregation
        
        # '''Ensemble Distillation'''
        # global_update = GlobalUpdateEns(args=args, dataset=dataset_unlabeled_train, 
        #                                val_data=dataset_val, selected_clients=selected_locals, net=net_glob)
        # global_update.train(net_glob, args)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter
            
            logging.info('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}, \
                         Best Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test, best_acc))

            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

        if (iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, '{}/best_{}.pt'.format(fed_name, iter + 1))
            model_save_path = os.path.join(base_dir, '{}/model_{}.pt'.format(fed_name, iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)

    logging.info('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))