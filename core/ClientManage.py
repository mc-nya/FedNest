import copy
from cv2 import log
import numpy as np

import torch
from torch.optim import SGD
from core.function import assign_hyper_gradient
from utils.Fed import FedAvg,FedAvgGradient, FedAvgP
from core.SGDClient import SGDClient
from core.SVRGClient import SVRGClient
from core.Client import Client

class ClientManage():
    def __init__(self,args, net_glob, client_idx, dataset, dict_users, hyper_param) -> None:
        self.net_glob=net_glob
        self.client_idx=client_idx
        self.args=args
        self.dataset=dataset
        self.dict_users=dict_users
           
        self.hyper_param = hyper_param
        self.hyper_optimizer=SGD([self.hyper_param[k] for k in self.hyper_param],
                                lr=self.args.hlr)


    def fed_in(self):
        print(self.client_idx)
        w_glob = self.net_glob.state_dict()
        if self.args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(self.args.num_users)]
        else:
            w_locals=[]

        loss_locals = []
        grad_locals = []
        client_locals = []

        for idx in self.client_idx:
            if self.args.optim == 'sgd':
                client = SGDClient(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
            elif self.args.optim == 'svrg':
                client = SVRGClient(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
                grad = client.batch_grad()
                grad_locals.append(grad)
            else:
                raise NotImplementedError
            client_locals.append(client)
        if self.args.optim == 'svrg':
            avg_grad = FedAvgGradient(grad_locals)
            for client in client_locals:
                client.set_avg_q(avg_grad)
        for client in client_locals:
            w, loss = client.train_epoch()
            if self.args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        self.net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        return w_glob, loss_avg

    def fedIHGP(self,client_locals):
        d_out_d_y_locals=[]
        for client in client_locals:
            d_out_d_y=client.grad_d_out_d_y()
            d_out_d_y_locals.append(d_out_d_y)
        p=FedAvgP(d_out_d_y_locals,self.args)
        
        p_locals=[]
        if self.args.hvp_method == 'global_batch':
            for i in range(self.args.neumann):
                for client in client_locals:
                    p_client = client.hvp_iter(p, self.args.hlr)
                    p_locals.append(p_client)
                p=FedAvgP(p_locals, self.args)
        elif self.args.hvp_method == 'local_batch':
            for client in client_locals:
                p_client=p.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                p_locals.append(p_client)
            p=FedAvgP(p_locals, self.args)
        elif self.args.hvp_method == 'seperate':
            for client in client_locals:
                d_out_d_y=client.grad_d_out_d_y()
                p_client=d_out_d_y.clone()
                for _ in range(self.args.neumann):
                    p_client = client.hvp_iter(p_client, self.args.hlr)
                p_locals.append(p_client)
            p=FedAvgP(p_locals, self.args)

        else:
            raise NotImplementedError
        #print("final p", p)
        return p

    def fed_out(self):
        client_locals=[]
        #self.outer_optimizer=torch.optim.SGD(self.hyper_param, lr=0.001, momentum=0)
        for idx in self.client_idx:
            client= Client(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
            client_locals.append(client)
        #for client in client_locals:
        p = self.fedIHGP(client_locals)

        hg_locals =[]
        for client in client_locals:
            hg= client.hyper_grad(p.clone())
            hg_locals.append(hg)
        hg_glob=FedAvgP(hg_locals, self.args)
        #print("hyper lr", hg_glob)
        assign_hyper_gradient(self.hyper_param, hg_glob)
        self.hyper_optimizer.step()
        print("hyper params: ", self.hyper_param)
        
        return 

            


    
