
from math import floor
from matplotlib import scale
import numpy as np
import argparse
import importlib
import torch
import os
import sys
from src.utils.worker_utils import read_data
from src.utils.client_utils import generate_real_time_v1, generate_time
from src.utils.config_utils import NAME_MAP
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS
import argparse
# from src.controllers.server import FedIoTServer
from src.controllers.client import FedIoTClient 
import src.communication.comm_config as connConfig

def fl_experiment(args, fedServer):
    # config settings 
    fedServer.config_experiment(args)
    
    # transfer config info to clients
    fedServer.init_clients()

    # deploy data
    # fedServer.deploy_data()

    # test speed of each client
    fedServer.test_comm_speed()
    
    fedServer.deploy_data()
    fedServer.train()
    # fedServer.wait()

def client_main():
    fedClient = FedIoTClient()
    fedClient.connect2server()
    while fedClient.is_experiment_ongoing():
        fedClient.init_config()
        fedClient.config_experiment()
        fedClient.test_comm_speed()
        fedClient.deploy_data()
        fedClient.train()
    
    print("Experiment Ended.")
def server_main():
    # Modify parameters here

   
    SEED_MAX = 8
    

    args = {
        'num_epoch': 50,
        'batch_size': 24,
        'num_round': 200,
        'model': 'logistic',
        'update_rate': 0,
        'num_clients': 40,
        'dataset': 'mnist_niid1_7_0',
        'alpha': 1.82076158099787,
        'lr': 0.1,
        'wd': 0.001,
        'gpu': False,
        'noaverage': False,
        'experiment_folder': 'property',
        'is_sys_heter': True,
        'without_rp': False, ##
        'decay': 'soft', ## API ## 
        'test_num':2,
        'C': [] , # cost
        'budget': 0,
        'v': [], # intrinsic value
        'optim_method':'matlab',

    }
    sd = [20, 20, 20, 20] # number of rounds to repeat
    opt = 0 
    # to compare which property, 0 for budget, 1 for cost, 2 for intrinsic value
    # parameters of each property test can be modified


    N = int(args['num_clients'])
    experimentConfig = {
        'seedMax': SEED_MAX,
    }

    
    # change parameters of property here
    for time_seed in range(0,1):
        args['time_seed'] = time_seed
        sd = [20, 20, 20, 20]
        
        basev = np.random.exponential(100, size = 40)                 
        baseC = np.random.exponential(5, size = 40)

        if opt == 0:
            # continue
            print('>>>>>>>>>>>>>>>>>>>>>>>compare B')
            C = baseC
            v = basev
            for i in range(4):
                if i == 3:
                    args['budget'] = 100
                if i == 0:
                    args['budget'] = 0.5
                if i == 1:
                    args['budget'] = 10
                if i == 2:
                    args['budget'] = 50
                

                args['v'] = v
                args['C'] = C
                
                for seed in range(0,sd[i]):
                    args['seed'] = seed
                    fedServer = FedIoTServer(args)
                    fedServer.connect2clients()
                    fedServer.start_experiment()
                    fl_experiment(args, fedServer)
                    fedServer.end_experiment()
        if opt == 1:

            print('>>>>>>>>>>>>>>>>>>>>>>>compare C')
            args['budget'] = 15.0
            v = np.random.exponential(100, size = 40)
            base = np.random.exponential(5, size = 40)
            for i in range(4):
                if i == 0:
                    C = [x/10 for x in base]
                if i == 1:
                    C = [x for x in base]
                if i == 2:
                    C = [2*x for x in base]
                if i == 3:
                    C = [5*x for x in base]
                
                args['v'] = v
                args['C'] = C
                
                for seed in range(0,sd[i]):
                    args['seed'] = seed
                    fedServer = FedIoTServer(args)
                    fedServer.connect2clients()
                    fedServer.start_experiment()
                    fl_experiment(args, fedServer)
                    fedServer.end_experiment()
        
        if opt == 2:

            print('>>>>>>>>>>>>>>>>>>>>>>>compare V')
            C = np.random.exponential(10, size = 40)
            args['budget'] = 5.0
            base = np.random.exponential(10000, size = 40)
            for i in range(4):
                if i == 3:
                    v = [5*x for x in base]
                if i == 0:
                    v = [0.0] * N
                if i == 1:
                    v = [0.1*x for x in base]
                if i == 2:
                    v = base
                args['v'] = v
                args['C'] = C
                
                for seed in range(0,sd[i]):
                    args['seed'] = seed
                    fedServer = FedIoTServer(args)
                    fedServer.connect2clients()
                    fedServer.start_experiment()
                    fl_experiment(args, fedServer)
                    fedServer.end_experiment()

    print("Experiment Ended.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='server or client')
    args = parser.parse_args()
            
    if args.mode == 'server':
        from src.controllers.server import FedIoTServer
        server_main()
    elif args.mode == 'client':
        client_main()
    else:
        raise Exception("Wrong parser parameter!")
        
        
        







    