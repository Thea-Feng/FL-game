
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
        'experiment_folder': 'bench',
        'is_sys_heter': True,
        'without_rp': False, ##
        'decay': 'soft', ## API ## 
        'test_num':2,
        'C': [5.0] * N, # cost
        'budget': 15.0,
        'v': np.random.exponential(scale=100,size=40), # intrinsic value
        'optim_method':'matlab',

    }
    
    N = int(args['num_clients'])
    fedServer = FedIoTServer(args)
    fedServer.connect2clients()

    experimentConfig = {
        'seedMax': SEED_MAX,
    }
    for t_seed in range(3,10):
        args['time_seed'] = t_seed
        for seed in range(1, 1+SEED_MAX):
            args['seed'] = seed


            args['algo'] = NAME_MAP['game']
            fedServer.start_experiment()
            fl_experiment(args, fedServer)
            fedServer.end_experiment()

            args['algo'] = NAME_MAP['right']
            fedServer.start_experiment()
            fedServer.end_experiment()
            fl_experiment(args, fedServer)
            
            args['algo'] = NAME_MAP['uniform']
            fedServer.start_experiment()
            fl_experiment(args, fedServer)
            fedServer.end_experiment()

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
        
        
        




