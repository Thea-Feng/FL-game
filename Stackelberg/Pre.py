
import numpy as np
import argparse
import importlib
import torch
import os
import sys
from src.utils.worker_utils import read_data
from src.utils.client_utils import generate_real_time_v1
from src.utils.config_utils import NAME_MAP
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

    parser = argparse.ArgumentParser()

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
    parser.add_argument('--real_round',
                        help='number of rounds to cal qk;',
                        type=int,
                        default=100)
    parser.add_argument('--c0',
                        help='a0/b0 in optim object func',
                        default=0,
                        type=float)
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
    parser.add_argument('--test_num',
                        help='test num;',
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
    
    parser.add_argument('--time_flag',
                        help='time flag',
                        type=str,
                        default='random_expCas')
    
    parser.add_argument('--decay',
                        help='decay method',
                        type=str,
                        default='')

    parser.add_argument('--without_r',
                        help='add more information;',
                        type=str,
                        default='False')

    parser.add_argument('--grad_path',
                        help='grad gk file path',
                        type=str,
                        default='')

    # parser.add_argument('--sys_heter',
    #                     help='consider system heterogenity;',
    #                     default=True)
    # parser.add_argument('--time_flag',
    #                     help='consider random time/pos/neg;',
    #                     type=str,
    #                     default='pos')

    parser.add_argument('--log',
                        help='log folder name;',
                        type=str,
                        default='')

    parsed = parser.parse_args()
    options = parsed.__dict__

    # if options['algo'] == 'fedavg7':
    #     print(">>>>>>>> OURS 1")
    #     options['num_round'] =  options['num_round']*2
    
    # update options
    options['without_r'] = options['without_r'] == 'True'
    # options['grad_path'] = args['grad_path']
    # options['smooth_thd'] = args['smooth_thd']
    # options['smooth_delta'] = args['smooth_delta']
    options['sys_heter'] = args['sys_heter']
    options['time_seed'] = args['time_seed']
    # options['time_flag'] = args['time_flag']
    # options['without_r'] = args['without_r']
    # options['optim_method'] = args['optim_method']
    # options['test_num'] = args['test_num']
    options['noprint'] = args['noprint']

    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    ## generate time 
    # options['client_times'] = generate_time(100,options['time_seed'], options['time_flag'])
    options['client_times'] = generate_real_time_v1(100, SEED=options['time_seed'],
                                                            k=options['clients_per_round'],
                                                            is_software=True)

    
    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # print("Model first is ",options['model'])

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

def premain(args):
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options(args)
    options['test_num'] = 100 + args['sround']
    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info)

    trainer.get_num_label()

    trainer.train(args['qk'])
    trainer.save_log()
    return trainer.logDict

def main(args):
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options(args)
    
    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info)
    trainer.real_round = options['real_round']
    
    trainer.logDict['budget'] = args['budget']*1.0
    for i, c in enumerate(trainer.clients):
        c.ck = args['C'][i]
        c.vk = args['v'][i]
        trainer.logDict['ck'].append(c.ck)
        trainer.logDict['vk'].append(c.vk)
        
    if args['algo'] not in ['game1']:
            trainer.logDict['gk'] = args['gk']
    trainer.get_num_label()

    trainer.train()
    trainer.save_log()

    if args['algo'] in ['game1']:
        return trainer.logDict['gk']
    else:
        return None

'''
    ** Test Sys heter 
        - dataset synthetic_A1_B1_niid
'''
def getA(Log1, Log2):
    ret = []
    stop_round = [19, 29, 49]

    for j in range(10):
        for i in range(3):
            R = stop_round[i]
            alpha = (R+1)*(Log1[j]['loss'][R] - Log2[j]['loss'][R])
            A1, A2 = 0, 0
            pk1 = Log1[j]['pk']
            pk2 = Log2[j]['pk']
            qk1 = Log1[j]['qk']
            qk2 = Log2[j]['qk']
            Gk1 = Log1[j]['gk'][R]
            Gk2 = Log2[j]['gk'][R]

            for k in range(len(pk1)):
                A1 += (1-qk1[k])/qk1[k] * (pk1[k]* Gk1[k])**2
                A2 += (1-qk2[k])/qk2[k] * (pk2[k]* Gk2[k])**2
            if A1 == A2:
                print("There is an zero", A1, A2)
            alpha /= (A1 - A2)
            ret.append(alpha)

    avg_ret = []

    for i in range(3):
        R = stop_round[i]
        alpha = 0
        for j in range(10):
            alpha += (R+1)*(Log1[j]['loss'][R] - Log2[j]['loss'][R])
        alpha /= 10
        A1, A2 = 0, 0
        pk1 = Log1[0]['pk']
        pk2 = Log2[0]['pk']
        qk1 = Log1[0]['qk']
        qk2 = Log2[0]['qk']
        for j in range(10):
            Gk1 = Log1[j]['gk'][R]
            Gk2 = Log2[j]['gk'][R]
            for k in range(len(pk1)):
                A1 += (1-qk1[k])/qk1[k] * (pk1[k]* Gk1[k])**2
                A2 += (1-qk2[k])/qk2[k] * (pk2[k]* Gk2[k])**2
        A1 /= 10
        A2 /= 10

        if A1 == A2:
            print("There is an zero")
        alpha /= (A1 - A2)
        avg_ret.append(alpha)
    
    return ret, avg_ret

if __name__ == '__main__':

    args = {}

    args['noprint'] = True
    args['sys_heter'] = True
    args['model'] = 'logistic'
    args['budget'] = 10   
    args['log'] = 'tmp' # where to save you result

    N = 40  # number of clients
    C = np.random.exponential(3, N)
    v = np.random.rand(100)
    for i in range(100):
        if i < 20:
            v[i] *= 5
        else:
            v[i] *= 100

    for time_seed in range(0,1):
        args['time_seed'] = time_seed

        for seed in range(0,1):
            args['seed'] = "{}".format(seed) 


            Log1 = []
            Log2 = []
            for i in range(10):
                args['algo'] = NAME_MAP['pretrain']
                args['sround'] = i
                args['qk'] = 0.1
                args['update_fre'] = '1'
                log = premain(args)
                Log1.append(log)
            
            for i in range(10):
                args['algo'] = NAME_MAP['pretrain']
                args['qk'] = 0.3
                args['sround'] = i + 10
                args['update_fre'] = '1'
                log = premain(args)
                Log2.append(log)
            ret, avg_ret = getA(Log1, Log2)
            print(ret, avg_ret)
            




