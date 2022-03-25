#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import yaml
import time
from core.test import test_img
from utils.Fed import FedAvg, FedAvgGradient
from models.SvrgUpdate import LocalUpdate
from utils.options import args_parser
from utils.dataset_normal import load_data
from models.ModelBuilder import build_model
from core.minmax.ClientManage_mm import ClientManageMM
from utils.logging import Logger
from core.function import assign_hyper_gradient
from torch.optim import SGD
import torch
import sys
import numpy as np
import copy

start_time = int(time.time())

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.model="linear"
    args.dataset = "minmax_synthetic"
    args.d=10
    args.n=10
    
    dataset_train, dataset_test, dict_users, args.img_size, dataset_train_real = load_data(args)
    net_glob = build_model(args)
    print(net_glob.x, net_glob.y_header)
    
    # copy weights
    w_glob = net_glob.state_dict()
    if args.output == None:
        logs = Logger(f'./save/Mar9/minmax_{args.optim}_{args.dataset}\
_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}_\
{args.lr}_blo{not args.no_blo}_\
IE{args.inner_ep}_N{args.neumann}_HLR{args.hlr}_{args.hvp_method}_{start_time}.yaml')  
    else:
        logs = Logger(args.output)                                                           
    
    hyper_param= [k for n,k in net_glob.named_parameters() if not "header" in n]
    param= [k for n,k in net_glob.named_parameters() if "header" in n]
    #print([n for n,k in net_glob.named_parameters()])
    # hyper_param= [net_glob.x]
    # param = [net_glob.y_header]
    #print(hyper_param,param)
    comm_round=0
    hyper_optimizer=SGD(hyper_param, lr=1)
    
    
    for iter in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        print("x and y", net_glob.x,net_glob.y_header)
        
        for _ in range(args.inner_ep):
            client_idx = np.random.choice(range(args.num_users), m, replace=False)
            client_manage=ClientManageMM(args,net_glob,client_idx, dataset_train, dict_users,hyper_param)
            w_glob, loss_avg = client_manage.fed_in()
            #print(comm_round)
            if args.optim == 'svrg':
                comm_round+=2
            else:
                comm_round+=1
            #print(comm_round)
        net_glob.load_state_dict(w_glob)
        
        if args.no_blo== False:
            client_idx = np.random.choice(range(args.num_users), m, replace=False)
            client_manage=ClientManageMM(args,net_glob,client_idx, dataset_train, dict_users,hyper_param)
            #print("hyper params: ", hyper_param)
            hg_glob, r = client_manage.fed_out()
                    #print("hyper lr", hg_glob)
            assign_hyper_gradient(hyper_param, hg_glob)
            hyper_optimizer.step()
            # print("hyper params: ", hyper_param)
            # print("params: ", param)

            comm_round+=r
        

        # print loss
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        # testing
        # net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train_real, args)
        # acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # print("Test acc/loss: {:.2f} {:.2f}".format(acc_test, loss_test),
        #       "Train acc/loss: {:.2f} {:.2f}".format(acc_train, loss_train),
        #       f"Comm round: {comm_round}")
        loss_x = torch.norm(net_glob.x).detach().cpu().numpy()**2
        loss_y = torch.norm(net_glob.y_header).detach().cpu().numpy()
        print("Loss: {:.2f} {:.2f}".format(loss_x,loss_y))
        #sum_loss = torch.norm(net_glob.x).detach().cpu().numpy()+torch.norm(net_glob.y_header).detach().cpu().numpy()
        logs.logging(client_idx, loss_x , loss_y , 0, 0, comm_round)
    
        if args.round>0 and comm_round>args.round:
            break
        
    logs.save()