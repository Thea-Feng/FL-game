
import numpy as np
import argparse
import importlib
import torch
import os
import sys
from src.utils.worker_utils import read_data
from src.utils.client_utils import generate_real_time_v1, generate_time
from src.utils.config_utils import NAME_MAP
from src.utils.alpha_utils import ALPHA_MAP
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


def read_options(args):

    sys.argv.extend(['--log', args['log']])

    # sys.argv.extend(['--sys_heter', args['sys_heter']])
    # sys.argv.extend(['--time_flag', args['time_flag']])
    sys.argv.extend(['--algo', args['algo']])
    # sys.argv.extend(['--dataset', args['dataset']])
    sys.argv.extend(['--model', args['model']])
    # sys.argv.extend(['--device', args['device']])
    # sys.argv.extend(['--c0', args['c0']])
    # sys.argv.extend(['--num_round', args['num_round']])
    # sys.argv.extend(['--num_epoch', args['num_epoch']])
    # sys.argv.extend(['--clients_per_round', args['clients_per_round']])
    sys.argv.extend(['--update_fre', args['update_fre']])
    sys.argv.extend(['--seed', args['seed']])
    # sys.argv.extend(['--time_seed', args['time_seed']])
    sys.argv.extend(['--num_clients', args['num_clients']])


    parser = argparse.ArgumentParser()
    parser.add_argument('--real_round',
                        help='number of rounds to cal qk;',
                        type=int,
                        default=100)
    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg4')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=True,
                        help='use gpu (default: False)')
    # parser.add_argument('--noprint',
    #                     action='store_true',
    #                     default=False,
    #                     help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--c0',
                        help='a0/b0 in optim object func',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=24)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
    parser.add_argument('--update_fre',
                        help='For Bing scheme 2, frequency of update',
                        type=int,
                        default=10)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--num_clients',
                        help='number of clients;',
                        type=int,
                        default=0)
    parser.add_argument('--test_num',
                        help='test number;',
                        type=int,
                        default=0)
    # parser.add_argument('--time_seed',
    #                     help='time seed for randomness;',
    #                     type=int,
    #                     default=0)

    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    
    parser.add_argument('--decay',
                        help='decay method',
                        type=str,
                        default='')

    parser.add_argument('--without_r',
                        help='add more information;',
                        type=str,
                        default='False')

    # parser.add_argument('--sys_heter',
    #                     help='consider system heterogenity;',
    #                     default=True)
    parser.add_argument('--time_flag',
                        help='consider random time/pos/neg;',
                        type=str,
                        default='pos')

    parser.add_argument('--log',
                        help='log folder name;',
                        type=str,
                        default='')
    parser.add_argument('--grad_path',
                        help='grad gk file path',
                        type=str,
                        default='')
    parsed = parser.parse_args()
    options = parsed.__dict__

    # update options
    options['without_r'] = options['without_r'] == 'True'
    # options['grad-path'] = args['grad-path']

    # options['smooth_thd'] = args['smooth_thd']
    # options['smooth_delta'] = args['smooth_delta']
    options['sys_heter'] = args['sys_heter']
    options['time_seed'] = args['time_seed']
    # options['time_flag'] = args['time_flag']
    # options['without_r'] = args['without_r']
    options['optim_method'] = args['optim_method']

    options['noprint'] = args['noprint']

    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    # if options['algo'] == 'fedavg7':
        # options['num_round'] += 400

    ## generate time 
    assert options['num_clients'] > 0

    # options['client_times'] = generate_time(options['num_clients'],options['time_seed'], options['time_flag'])
    options['client_times'] = generate_real_time_v1(options['num_clients'], SEED=options['time_seed'],
                                                            k=options['clients_per_round'],
                                                            is_software=False)
    
    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Load selected trainer
    trainer_path = 'src.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        if not (keyPair[0] in ['client_times']):
            print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


def main(args):
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options(args)
    options['test_num']  += args['sround']

    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info)
    trainer.real_round = options['real_round']
    
    trainer.logDict['budget'] = args['budget']*1.0
    # print(args)
    trainer.logDict['alpha'] = trainer.alpha = ALPHA_MAP[options['dataset']]

    for i, c in enumerate(trainer.clients):
        c.ck = args['C'][i]
        c.vk = args['v'][i]
        trainer.logDict['ck'].append(c.ck)
        trainer.logDict['vk'].append(c.vk)
        
    if args['algo'] not in ['game1']:
            trainer.logDict['gk'] = args['gk']
    trainer.train()
    trainer.save_log()
    if args['algo'] in ['game1']:
        return trainer.logDict['gk']
    else:
        return None
'''
    ** Test Sys heter
        - 30 clients  
        - dataset -> mnist_niid_1_7
        - time seed 200-207
        - seed      1-5
'''


if __name__ == '__main__':

    args = {}

    args['noprint'] = True
    args['sys_heter'] = True
    args['model'] = 'logistic'
    args['optim_method'] = 'matlab'

   
    # args that you can modify
    args['log'] = 'tmp' # where to save your result
    args['num_clients'] = '40' # number of clients
    sd = [20, 20, 20, 20] # number of rounds to repeat
   
    opt = 0 
    # to compare which property, 0 for budget, 1 for cost, 2 for intrinsic value
    # parameters of each property test can be modified


    N = int(args['num_clients'])
    for time_seed in range(0,1):
        args['time_seed'] = time_seed

        if opt == 0:
            # Here is an expample how parameters are set,
            # for C and V, users can specify by themselves
            print('>>>>>>>>>>>>>>>>>>>>>>>compare B')
            C = np.random.exponential(5, size = 40)
            v = np.random.exponential(100, size = 40)                 
            for i in range(4):
                if i == 0:
                    args['budget'] = 1
                if i == 1:
                    args['budget'] = 10
                if i == 2:
                    args['budget'] = 50
                if i == 3:
                    args['budget'] = 200

                for seed in range(0,sd[i]):
                    args['seed'] = "{}".format(seed) 
                    args['algo'] = NAME_MAP['game']
                    args['update_fre'] = '1'
                    args['C'] = C
                    args['v'] = v
                    args['gk'] = None
                    args['sround'] = i*1000+seed
                    gk = main(args)

        if opt == 1:

            print('>>>>>>>>>>>>>>>>>>>>>>>compare C')
            args['budget'] = 15.0
            v = 

            for i in range(4):
                if i == 0:
                    C = 
                if i == 1:
                    C = 
                if i == 2:
                    C = 
                if i == 3:
                    C = 
                
                for seed in range(0,sd[i]):
                    args['seed'] = "{}".format(seed) 
                    args['algo'] = NAME_MAP['game']
                    args['update_fre'] = '1'
                    args['C'] = C
                    args['v'] = v
                    args['sround'] = i*1000+seed
                    gk = main(args)
        
        if opt == 2:

            print('>>>>>>>>>>>>>>>>>>>>>>>compare V')
            C = 
            args['budget'] = 

            for i in range(4):
                if i == 0:
                    v = 
                if i == 1:
                    v = 
                if i == 2:
                    v = 
                if i == 3:
                    v = 
                
                for seed in range(0,sd[i]):
                    args['seed'] = "{}".format(seed) 
                    args['algo'] = NAME_MAP['game']
                    args['update_fre'] = '1'
                    args['C'] = C
                    args['v'] = v
                    args['sround'] = i*1000+seed
                    gk = main(args)