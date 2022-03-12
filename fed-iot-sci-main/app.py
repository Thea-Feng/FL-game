# Randy Xiao 2021-6-09
# debug version

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

# 
def server_main():
    # Modify parameters here

    NUM_OF_CLIENTS = 40
    SEED_MAX = 8
    EXPERIMENT_FOLDER = '6-25'

    

    args = {
        'num_epoch': 50,
        'batch_size': 24,
        'num_round': 200,
        'model': 'logistic',
        'update_rate': 0,
        'num_clients': NUM_OF_CLIENTS,
        'dataset': 'mnist_niid1_7_0',
        'lr': 0.1,
        'wd': 0.001,
        'gpu': False,
        'noaverage': False,
        'experiment_folder': EXPERIMENT_FOLDER,
        'is_sys_heter': True,
        'eval_every': 5,
        'c0': 5,    ##
        'without_rp': False, ##
        'decay': 'inverse', ## API ## 
        'test_num':2,
    }

    fedServer = FedIoTServer(args)
    fedServer.connect2clients()

    experimentConfig = {
        'seedMax': SEED_MAX,
    }
    for t_seed in range(3,10):
        args['time_seed'] = t_seed
        for seed in range(1, 1+SEED_MAX):
            args['seed'] = seed

            # lx scheme 1
            args['algo'] = 'game1'
            args['update_fre'] = 0
            args['num_round'] = 300
            fedServer.start_experiment()
            fl_experiment(args, fedServer)

        #     # lx scheme 2t
        #     args['algo'] = 'lx_s2t'
        #     args['update_fre'] = 0
        #     args['num_round'] = 300
        #     fedServer.start_experiment()
        #     fl_experiment(args, fedServer)

        #     # bing scheme 1
        #     args['algo'] = 'bing_s1'
        #     args['update_fre'] = 0
        #     args['num_round'] = 200
        #     fedServer.start_experiment()
        #     fl_experiment(args, fedServer)

        #     # bing gk
        #     args['algo'] = 'bing_gk'
        #     args['update_fre'] = 0
        #     args['num_round'] = 200
        #     fedServer.start_experiment()
        #     fl_experiment(args, fedServer)

        #     # bing scheme 2 10
        #     args['algo'] = 'bing_s2'
        #     args['update_fre'] = 10
        #     args['num_round'] = 200
        #     fedServer.start_experiment()
        #     fl_experiment(args, fedServer)

        #     # bing scheme 2 50
        #     args['algo'] = 'bing_s2'
        #     args['update_fre'] = 50
        #     args['num_round'] = 200
        #     fedServer.start_experiment()
        #     fl_experiment(args, fedServer)

        #     # bing scheme 2 300
        #     args['algo'] = 'bing_s2'
        #     args['update_fre'] = 300
        #     args['num_round'] = 200
        #     fedServer.start_experiment()
        #     fl_experiment(args, fedServer)

        #     # bing scheme 3
        #     args['algo'] = 'bing_s3'
        #     args['update_fre'] = 0
        #     args['num_round'] = 200
        #     fedServer.start_experiment()
        #     fl_experiment(args, fedServer)

        # # full sample
        # args['algo'] = 'full'
        # args['update_fre'] = 0 
        # args['seed'] = 0
        # fedServer.start_experiment()
        # fl_experiment(args, fedServer)

    fedServer.end_experiment()

    print("Experiment Ended.")

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
    

if __name__ == "__main__":
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
