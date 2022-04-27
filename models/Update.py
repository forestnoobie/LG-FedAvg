#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
import copy
import math
import pdb

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    
class BroadcastHetero(object):
    # Use private dataset to distll from server
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.KLDivLoss(reduction='batchmean')
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=4)
 
    def train(self, net_local, net_global, args):
        net_local.train()
        net_global.eval()
        # train and update
        optimizer = torch.optim.SGD(net_local.parameters(), lr=args.local_broadcast_lr, momentum=0.5)
        epoch_loss = []
        T = self.args.broadcast_t
        local_eps = self.args.local_broadcast_ep
        
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net_local.zero_grad()
                output = net_local(images)
                output_glob = net_global(images)

                log_probs = F.log_softmax(output/T, dim=1)
                glob_probs = F.sotmax(output_glob/T, dim=1)
                loss = self.loss_func(log_probs, glob_probs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        logging.info("Epoch loss : {}".format(epoch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class GlobalUpdateEnsHetero(object):
    def __init__(self, args, dataset=None, val_data=None, idx_users=None, dict_users_model=None):
        self.args = args
        self.loss_func = nn.KLDivLoss(reduction='batchmean')
        self.client_models = [dict_users_model[idx] for idx in idx_users]
        self.ldr_train = DataLoader(dataset, batch_size=self.args.unlabeled_batch_size, shuffle=True, num_workers=4)
        self.ldr_val = DataLoader(val_data, batch_size=self.args.local_bs, shuffle=False, num_workers=4)

    
    def get_avg_logits(self, images):
        data_num = images.size(0)
        model = copy.deepcopy(self.model)
        avg_logits = torch.zeros(data_num, self.args.num_classes, device=self.args.device)
        
        with torch.no_grad():
            for client_model in self.client_models:
                client_model.eval()
                images = images.to(self.args.device)
                client_model = client_model.to(self.args.device)
                avg_logits += client_model(images)
        avg_logits /= len(self.client_models)
        avg_logits = F.softmax(avg_logits, dim=1)
        return avg_logits.detach()
    
    def train(self, net, args):
        # Ensemble update
        net.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                     lr=args.server_lr, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, args.server_steps)

        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        epoch_loss = []
        
        while curr_step < args.server_steps and patience_step < args.server_patience_steps:
            with tqdm(self.ldr_train, unit="Step") as tstep:
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")            
                    if curr_step < args.server_steps and patience_step < args.server_patience_steps:
                        net.train()
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net.zero_grad()
                        output = net(images)
                        log_probs = F.log_softmax(output, dim=1)
                        avg_probs = self.get_avg_logits(images)

                        loss = self.loss_func(log_probs, avg_probs)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        batch_loss.append(loss.item())
                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if self.ldr_val:
                            if curr_step % 100 == 0 :
                                curr_val_acc = self.validate(net, self.ldr_val, self.args.device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0

                        tstep.set_postfix(val_acc=curr_val_acc, best_val_acc=best_val_acc, step_loss=loss.item())
                    else :
                        break


                epoch_loss.append(sum(batch_loss)/len(batch_loss))
       
        logging.info("Best validation : {}".format(best_val_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
    def validate(self, net, val_data, device, args):
        net.eval()
        correct = 0
        n_data = 0

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(val_data):
                if type(x) == int or type(labels) == int : import ipdb; ipdb.set_trace(context=15)
                x, labels = x.to(device), labels.to(device)
                output = net(x)
                predicted = torch.argmax(output, dim=1)

                batch_correct = predicted.eq(labels).sum()
                batch_num = x.size(0)

                correct += batch_correct
                n_data += batch_num

            val_acc = correct / n_data
        return val_acc.detach().item()

    
    
class GlobalUpdateEns(object):
    def __init__(self, args, dataset=None, val_data=None, selected_clients=None, net=None):
        self.args = args
        self.loss_func = nn.KLDivLoss(reduction='batchmean')
        self.selected_clients = selected_clients
        self.ldr_train = DataLoader(dataset, batch_size=self.args.unlabeled_batch_size, shuffle=True, num_workers=4)
        self.ldr_val = DataLoader(val_data, batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.model = copy.deepcopy(net)
        self.load_client_model()
        
    def load_client_model(self):
        model = copy.deepcopy(self.model)
        client_models = []
        with torch.no_grad():
            for w_local in self.selected_clients:
                model.cpu().load_state_dict(w_local)
                model.eval()
                client_models.append(copy.deepcopy(model))
        
        self.client_models = client_models
    
    def get_avg_logits(self, images):
        data_num = images.size(0)
        model = copy.deepcopy(self.model)
        avg_logits = torch.zeros(data_num, self.args.num_classes, device=self.args.device)
        
        with torch.no_grad():
            for client_model in self.client_models:
                client_model.eval()
                images = images.to(self.args.device)
                client_model = client_model.to(self.args.device)
                avg_logits += client_model(images)
        avg_logits /= len(self.client_models)
        avg_logits = F.softmax(avg_logits, dim=1)
        return avg_logits.detach()
    
    def train(self, net, args):
        # Ensemble update
        net.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                     lr=args.server_lr, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, args.server_steps)

        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        epoch_loss = []
        
        while curr_step < args.server_steps and patience_step < args.server_patience_steps:
            with tqdm(self.ldr_train, unit="Step") as tstep:
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")            
                    if curr_step < args.server_steps and patience_step < args.server_patience_steps:
                        net.train()
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net.zero_grad()
                        output = net(images)
                        log_probs = F.log_softmax(output, dim=1)
                        avg_probs = self.get_avg_logits(images)

                        loss = self.loss_func(log_probs, avg_probs)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        batch_loss.append(loss.item())
                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if self.ldr_val:
                            if curr_step % 100 == 0 :
                                curr_val_acc = self.validate(net, self.ldr_val, self.args.device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0

                        tstep.set_postfix(val_acc=curr_val_acc, best_val_acc=best_val_acc, step_loss=loss.item())
                    else :
                        break


                epoch_loss.append(sum(batch_loss)/len(batch_loss))
       
        logging.info("Best validation : {}".format(best_val_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
    def validate(self, net, val_data, device, args):
        net.eval()
        correct = 0
        n_data = 0

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(val_data):
                if type(x) == int or type(labels) == int : import ipdb; ipdb.set_trace(context=15)
                x, labels = x.to(device), labels.to(device)
                output = net(x)
                predicted = torch.argmax(output, dim=1)

                batch_correct = predicted.eq(labels).sum()
                batch_num = x.size(0)

                correct += batch_correct
                n_data += batch_num

            val_acc = correct / n_data
        return val_acc.detach().item()
        
    

        
        
        
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True, num_workers=4)
        self.pretrain = pretrain

    def train(self, net, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                output = net(images)

                loss = self.loss_func(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True, num_workers=4)
        self.pretrain = pretrain

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                W = W_glob.clone()

                W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, idx] = W_local

                loss_regularizer = 0
                loss_regularizer += W.norm() ** 2

                k = 4000
                for i in range(W.shape[0] // k):
                    x = W[i * k:(i+1) * k, :]
                    loss_regularizer += x.mm(omega).mm(x.T).trace()
                f = (int)(math.log10(W.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss = loss + loss_regularizer
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
