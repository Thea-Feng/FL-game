# Randy Xiao 2021-6-09

import os
import torch
import numpy as np
from torch import Tensor
from src.communication.comm_tcp import TCP_SOCKET
import src.communication.comm_config as connConfig
from src.optimizers.gd import GD
from src.models.client import Client
from src.models.model import choose_model
from src.models.worker import LrdWorker, LrAdjustWorker
from src.utils.worker_utils import read_data


class FedIoTClient:

    def __init__(self, ):

        self.cid = None 
        self.IP = connConfig.IP 
        self.PORT = 1234
        self.args = None 
        self.commLimit = 1   # bandwith limitation 
        self.compLimit = 1
        self.debug = True 

    def connect2server(self, ):
        # connect to the server
        # don't disconnect until the end of the entire experiments
        self.socketClient = TCP_SOCKET(self.IP, self.PORT)
        self.socketClient.setupClient()


    def deploy_data(self, ):
        # recire data if unavailable
        # ortherwise use local data
        # assign cid, commLimit, compLimit
        idx = self.dataset_info.find("_")
        if idx != -1:
            dataset_name, sub_data = self.dataset_info[:idx], self.dataset_info[idx+1:]
        else:
            dataset_name, sub_data = self.dataset_info, None

        # data path
        train_path = os.path.join('./data', dataset_name, 'data', 'train')
        test_path = os.path.join('./data', dataset_name, 'data', 'test')

        # fetech data 
        all_data_info = read_data(train_path, test_path, sub_data)
        
        users, groups, train_data, test_data = all_data_info

        message, info = self.socketClient.recv_msg(connConfig.SERVER_SENDER)
        if self.debug:
            print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
        
        # print(message)
        msg = message['msg']
        self.cid = msg['cid']
        self.commLimit = msg['commLimit']
        self.compLimit = msg['compLimit']
        
        user= users[self.cid]
        self.client = Client(self.cid, None, train_data[user], test_data[user], self.batch_size, self.worker)

        message, info = self.socketClient.recv_msg(connConfig.SERVER_SENDER)
        if self.debug:
            print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
        
        msg = message['msg']
        self.all_train_data_num = msg['all_train_data_num']


    def is_experiment_ongoing(self, ):
        message, info = self.socketClient.recv_msg(connConfig.SERVER_SENDER)
        return message['type'] == connConfig.MSG_EXP_START

    def init_config(self, ):
        message, info = self.socketClient.recv_msg(connConfig.SERVER_SENDER)
        # print(message['type'])
        self.args = message['msg'] 
        # print("config",self.args)
        if self.debug:
            print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))

    def config_experiment(self, ):
        self.algo = self.args['algo']
        self.model = self.args['model']
        self.dataset_info = self.args['dataset']
        self.seed = self.args['seed']
        self.batch_size = self.args['batch_size']
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

        # Train
        model = choose_model(self.args)
        self.optimizer = GD(model.parameters(), lr=self.args['lr'], weight_decay=self.args['wd'])
        self.num_epoch = self.args['num_epoch']

        self.worker = LrAdjustWorker(model, self.optimizer, self.args)


    def test_comm_speed(self, ):
        for exp in range(connConfig.COMM_TEST_EXP):
            message, info = self.socketClient.recv_msg(connConfig.SERVER_SENDER)
            # print("done")
            # print(message['type'])
            model = message['msg'] 
            if self.debug:
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))

    def test_comp_speed(self, ):
        pass 

    
    def local_train(self, dict):
        # recive signal and 
        trainConfig={
            'latest_model_tensor': Tensor(dict['latest_model_list']),
            'comp_lim': self.compLimit,
            'lr': dict['lr']
        }
        
        local_train_result = self.client.local_train(trainConfig)
        self.socketClient.recv_signal(connConfig.SERVER_SENDER)
        info = self.socketClient.send_msg(
            message={
                'type': connConfig.MSG_TRAIN_RESULT,
                'msg': local_train_result,
                'tag': '-'*self.commLimit*connConfig.TAG_SIZE,
            },
            reciver=connConfig.SERVER_SENDER
        )
        if self.debug:
            print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
            print("local train finished")
        
        # if self.args['decay'] == 'inverse':
        #     self.optimizer.inverse_prop_decay_learning_rate(dict['round_i'])
        # elif self.args['decay'] == 'soft':
        #     self.optimizer.soft_decay_learning_rate()
        # else:
        #     raise Exception("Wrong DECAY Method!")

    def get_grad(self, dict):
        latest_model = Tensor(dict['latest_model_list'])
        self.client.set_flat_model_params(latest_model)
        print('->')
        c_grad = self.client.get_grad()
        print("grad", c_grad)
        info = self.socketClient.send_msg(
            message={
                'type': connConfig.MSG_GET_GRAD,
                'msg': {
                    'c_grad': c_grad.item(),
                },
                'tag': ''
            },
            reciver=connConfig.SERVER_SENDER,
        )
        if self.debug:
            print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))


    def train(self, ):
        while True:
            inst, info = self.socketClient.recv_msg(connConfig.SERVER_SENDER)
            if self.debug:
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
            
            if inst['type'] == connConfig.MSG_END_EXPERIMENT:
                if self.debug:
                    print("Train Ended.")
                return 
            elif inst['type'] == connConfig.MSG_LOCAL_TRAIN:
                if self.debug:
                    print("Local Train.")
                self.local_train(inst['msg'])
            elif inst['type'] == connConfig.MSG_GET_GRAD:
                if self.debug:
                    print("Get Grad")
                self.get_grad(inst['msg'])
                
            else:
                raise Exception("Unknown Instruction")

