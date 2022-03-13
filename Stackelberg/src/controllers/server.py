# Randy Xiao 2021-6-09

import os
from src.trainers.game1 import Game1Trainer
from src.trainers.Rightgame import RightgameTrainer
from src.trainers.Unigame import UnigameTrainer

import time
import json
import torch
import numpy as np
from src.optimizers.gd import GD
from src.models.model import choose_model
from src.models.client import Client
from src.communication.comm_tcp import TCP_SOCKET
from src.communication.comm_tools import get_ip
import src.communication.comm_config as connConfig
from src.utils.worker_utils import read_data
from src.models.worker import LrdWorker, LrAdjustWorker


class FedIoTServer:

    def __init__(self, args):
        # 
        self.num_clients = args['num_clients']
        self.IP = connConfig.IP
        self.PORT = 1234
        self.debug = False 

    def config_experiment(self, args):
        self.args = args 
        self.algo = args['algo']
        self.model = args['model']
        self.dataset_info = args['dataset']
        self.seed = args['seed']
        self.batch_size = args['batch_size']
        self.all_train_data_num = 0
        self.time_seed = args['time_seed']
        np.random.seed(1 + self.seed)
        torch.manual_seed(12 + self.seed)

        dataset = self.dataset_info.split('_')[0]
        if dataset == 'synthetic':
            self.args.update({'input_shape': 60, 'num_class': 10})
        elif dataset == 'mnist' or dataset == 'nist':
            if self.model == 'logistic' or self.model == '2nn':
                self.args.update({'input_shape': 784, 'num_class': 10}) 
            else:
                self.args.update({'input_shape': (1, 28, 28), 'num_class': 10})
        elif dataset == 'emnist':
            if model == 'logistic':
                self.args.update({'input_shape': 784, 'num_class': 26})
            else:
                self.args.update({'input_shape':(1,28,28), 'num_class':26})


        # Trainer
        model = choose_model(self.args)
        self.optimizer = GD(model.parameters(), lr=self.args['lr'], weight_decay=self.args['wd'])
        self.num_epoch = self.args['num_epoch']
        self.worker = LrAdjustWorker(model, self.optimizer, self.args)

        self.clients = []

    def connect2clients(self, ):
        # connect to all clients
        print("default IP->{}, PORT->{}".format(self.IP, self.PORT))
        print("current IP->{}, PORT->{}".format(get_ip(), self.PORT))
        self.socketServer = TCP_SOCKET(self.IP, self.PORT)
        self.socketServer.setupServer()
        self.socketServer.connect2Clients(self.num_clients)

    def start_experiment(self, ):
        for i in range(self.num_clients):
            info = self.socketServer.send_msg(
                message={
                    'type': connConfig.MSG_EXP_START, 'msg':'', 'tag':''}, 
                reciver={
                    'type': 'client',
                    'num': i, }
            )

    def end_experiment(self, ):
        for i in range(self.num_clients):
            info = self.socketServer.send_msg(
                message={
                    'type': connConfig.MSG_EXP_END, 'msg':'', 'tag':''}, 
                reciver={
                    'type': 'client',
                    'num': i, }
            )

    def init_clients(self, ):
        # transfer config info to clients

        for i in range(self.num_clients):
            info = self.socketServer.send_msg(
                message={
                    'type': connConfig.MSG_CONFIG,
                    'msg': self.args,
                    'tag': ''}, 
                reciver={
                    'type': 'client',
                    'num': i, }
            )
            if self.debug:
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))

    def deploy_data(self, distMode='random'):
        # send data to each client
        # client index is based on clients' profile

        '''
            client_times ?
        '''
        idx = self.dataset_info.find("_")
        if idx != -1:
            dataset_name, sub_data = self.dataset_info[:idx], self.dataset_info[idx+1:]
        else:
            dataset_name, sub_data = self.dataset_info, None

        # data path
        train_path = os.path.join('./data', dataset_name, 'data', 'train')
        test_path = os.path.join('./data', dataset_name, 'data', 'test')
        # print(train_path, test_path, sub_data)
        # fetech data 
        all_data_info = read_data(train_path, test_path, sub_data)
        
        users, groups, train_data, test_data = all_data_info

        print("debug: users->{}, groups->{}".format(users, groups))

        # consider all client have the dataset file,
        # here only assign client id
        IndexOfClients = [ i for i in range(self.num_clients)]
        # commDistOfClients = [ i for i in range(self.num_clients)]
        compDistOfClients = [ 1 for i in range(self.num_clients)]


        for i, tComm, tComp in zip(IndexOfClients, self.comm_lim, compDistOfClients):
            info = self.socketServer.send_msg(
                message={
                    'type': connConfig.MSG_DEPLOY_DATA,
                    'msg': {
                        'cid': i, 
                        'commLimit': tComm,
                        'compLimit': tComp,
                    },
                    'tag': ''
                },
                reciver={
                    'type': connConfig.CLIENT,
                    'num': i
                }
            )
            if self.debug:
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
            # print(i,users)
            user= users[i]

            self.all_train_data_num += len(train_data[user])
            self.clients.append(
                Client(i, None, train_data[user], test_data[user], self.batch_size, self.worker)
            )

        # self.client_times = np.array(compDistOfClients) + np.array(commDistOfClients)

        for i in range(self.num_clients):
            print(i)
            info = self.socketServer.send_msg(
                message={
                    'type': connConfig.MSG_DEPLOY_DATA,
                    'msg': {
                        'all_train_data_num': self.all_train_data_num,
                    },
                    'tag': ''
                },
                reciver={
                    'type': connConfig.CLIENT,
                    'num': i
                }
            )
            if self.debug:
                print("all train data")
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
   
    def test_comm_speed(self, ):

        # test communication speed of all clients

        '''
            !!! Change to recieve time !!!

        '''


        flatModel = self.worker.get_flat_model_params().detach()
        
        l = np.array(flatModel.tolist())
        l.tobytes()
        print("len", len(l))

        self.comm_lim = [i for i in range(self.num_clients)]
        np.random.seed(5 + self.time_seed)
        np.random.shuffle(self.comm_lim)
        np.random.seed(10 + self.seed)
        time_exp = []
        for exp in range(connConfig.COMM_TEST_EXP):
            print("Exp", exp)
            t_clients = []
            for i in range(self.num_clients):
                blim_comm = self.comm_lim[i]

                info = self.socketServer.send_msg(
                    message={
                        'type': connConfig.MSG_TEST_COMM_SPEED,
                        'msg': flatModel.tolist(),
                        'tag': '-'*connConfig.TAG_SIZE*i}, 
                    reciver={
                        'type': 'client',
                        'num': i, }
                )
                # if self.debug:
                #     print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
                t_clients.append(info['time'])
            time_exp.append(t_clients)
        # print("time list", time_exp)
        self.client_times = np.mean(time_exp,axis=0)
        with open('./log/3-2/time.json','w') as fp:
            json.dump({"time_exp":time_exp}, fp)


        # print(flatModel.tolist())

    def test_comp_speed(self, ):
        # test comp speed of all clients
        pass 

    def train(self, ):
        # perform Federated Learning
        # naming is consistent with the simulator
    
        trainerConfig={
            'worker': self.worker,
            'all_train_data_num': self.all_train_data_num,
            'clients': self.clients,
            'client_times': self.client_times,
            'socketServer': self.socketServer,
            'debug': self.debug,
        }

        if self.algo == 'game1':
            self.trainer = Game1Trainer(self.args, trainerConfig)
        if self.algo == 'right':
            self.trainer = RightgameTrainer(self.args, trainerConfig)
        elif self.algo == 'uniform':
            self.trainer = UnigameTrainer(self.args, trainerConfig)
        else:
            raise Exception("Algorithm {} is unavailable".format(self.algo))

        self.trainer.train()
    
    def wait(self, waitTime=-1):
        if waitTime == -1:
            while True:
                time.sleep(1)
        else:
            time.sleep(waitTime)
        
        


    