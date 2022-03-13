from posixpath import realpath
from pydoc import cli
from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import torch
import time 
import random
from scipy.optimize import minimize
# import numpy as np

criterion = torch.nn.CrossEntropyLoss()


# Bing  Scheme I & II

class Game1Trainer(BaseTrainer):

    def __init__(self, options, dataset):
        '''
        dataset
        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found     (what is groups)
            train_data: dictionary of train data (ndarray)
            test_data: dictionary of test data (ndarray)
        '''
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']
        worker = LrdWorker(model, self.optimizer, options)
        super(Game1Trainer, self).__init__(options, dataset, worker=worker)
        
    def calc_grad(self, ):
        for c in self.clients:
            c.set_flat_model_params(self.latest_model)
            c_grad = c.get_grad()
            c.Gk = c_grad.item()

    def pretrain(self):
        print(">>> pretrain to get Gk")
        self.latest_model = self.worker.get_flat_model_params().detach()
        st_ = time.time()
        
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        gk1 = []
        gk2 = []
        # gk3 = []
        for i, c in enumerate(self.clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)
            soln, stat = c.local_train()
            solns.append(soln)
            stats.append(stat)
            gk1.append(c.get_grad().item())

        # self.aggregate()
        clients = self.clients
        averaged_solution = torch.zeros_like(self.latest_model)
        for i, (num_sample, local_solution) in enumerate(solns):
            c = clients[i]
            averaged_solution += local_solution * c.pk
                             
        self.latest_model = averaged_solution.detach()

        # normal one
        for c in self.clients:
            c.set_flat_model_params(self.latest_model)
            c.Gk = c.get_grad()
            gk2.append(c.Gk.item())


        # #####fix C   
        # C = []
        # for c in self.clients:
        #     c.set_flat_model_params(self.latest_model)
        #     c.Gk = c.get_grad()
        #     gk2.append(c.Gk.item())
        #     c.ck = ((c.pk*c.Gk.item())**2)
        #     C.append(c.ck)


        self.logDict['gk'] = gk2
        for i, x in enumerate(gk2):
            pGi = self.logDict['pk'][i] * x
            self.logDict['pG'].append(pGi)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!! fix C
            # self.logDict['ck'] = C

        # print("gk1",gk1)
        # print("gk2",gk2)
        print("Pre train done | Spent %ss" %(time.time() - st_) )

    def train(self):
        
        # Get grad_norm
        self.pretrain()
        prob = self.compute_prob(duipai=False)
        # print('prob1',prob)
        print(np.array(prob.sum()/100))
        self.prob = prob

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! pay attention to here
        R = self.num_round
        # R = 1000.0

        self.latest_model = self.worker.get_flat_model_params().detach()

        tot = 0
        for i, c in enumerate(self.clients):
            c.round = np.random.choice([0, 1], size = self.num_round, p = [1-c.qk, c.qk])
            # random.shuffle(c.round)
            c.real = np.sum(c.round) / self.num_round
            self.logDict['qk'].append(c.qk)
            self.logDict['real_sample'].append(c.real)
            self.logDict['sample_diff'].append((c.real-c.qk)/c.qk)
            self.logDict['Pk'].append(2*c.ck*c.qk-c.vk*self.alpha/R*(c.pk**2)*(c.Gk**2)/(c.qk**2))
            # print("sample diff", i, (c.real-c.qk)/c.qk)
                   
            tot += c.qk *self.logDict['Pk'][i]

        print('here is budget compare', self.logDict['budget'], tot)


        for round_i in range(self.num_round):
            
            st_ = time.time()
            
            selected_clients = []
            select_index = []
            clients_cnt = 0
            for i, c in enumerate(self.clients):
                if c.round[round_i] == 1:
                    selected_clients.append(c)
                    select_index.append(c.cid)
                    clients_cnt += 1
                    

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)


            # Update latest model
            self.latest_model = self.aggregate(solns, clients=selected_clients)

            # diff decay method
            if self.decay == 'round':
                print("round decay")
                self.optimizer.inverse_prop_decay_learning_rate(round_i)
            else:
                print("soft decay")
                self.optimizer.soft_decay_learning_rate()
                
            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

            


            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)
            # log 
            self.logDict['sel_clients'].append([c.cid for c in selected_clients])
            # print(type(self.prob.ite))
            # self.logDict['qk'].append(self.prob.tolist())
            self.logDict['loss'].append(self.loss)
            self.logDict['acc'].append(self.acc)
            # self.logDict['gk'].append([c.Gk.item() for c in self.clients])
            self.logDict['lambda'] = [1/(4*R/self.alpha*c.ck*(c.qk**3)/((c.pk**2)*(c.Gk.item()**2))+c.vk) for c in self.clients]

            if self.isSysHeter:
                self.logDict['time'].append(self.calc_time(selected_clients))

                
            print("Round %s | Spent %ss" %(round_i, time.time() - st_) )
        self.save_log()

        self.end_train()
        # print('lam',self.logDict['lambda'])
        # print('Pk',self.logDict['Pk'])   
            
            

    def compute_prob(self, duipai=False):
        # here is real
        A = []
        B = []
        C = []
        for i, c in enumerate(self.clients):
            A.append((c.pk**2)*(c.Gk**2))
            B.append(c.vk*(c.pk**2)*(c.Gk**2))
            C.append(c.ck)
        if not duipai:
            probs = self.get_qk_matlab(A, B, C, self.logDict['budget'], self.alpha)

            for i, c in enumerate(self.clients):
                c.qk = probs[i]
        return probs

        

    def aggregate(self, solns, clients):
        averaged_solution = torch.zeros_like(self.latest_model)
        sub = 0
        # averaged_solution = np.zeros(self.latest_model.shape)
        
        w0 = self.latest_model

        assert len(solns) == len(clients)
        for i, (num_sample, local_solution) in enumerate(solns):
            c = clients[i]
            averaged_solution += local_solution * c.pk / c.qk
            sub += c.pk / c.qk

        # averaged_solution /= self.clients_per_round
        # sub /= self.clients_per_round
        
        averaged_solution = averaged_solution + w0*(1 - sub)

        return averaged_solution.detach()
