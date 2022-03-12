# GLOBAL PARAMETERS
DATASETS = ['sent140', 'nist', 'shakespeare',
            'mnist', 'synthetic', 'cifar10','emnist','femnist']
TRAINERS = {'fedavg': 'FedAvgTrainer',
            'fedavg4': 'FedAvg4Trainer',
            'fedavg5': 'FedAvg5Trainer',
            'fedavg6': 'FedAvg6Trainer',   # scheme I FG
            'fedavg6b': 'FedAvg6bTrainer', # scheme I SG
            'fedavg7': 'FedAvg7Trainer',    # Scheme I (preprocess grad)
            'fedavg8': 'FedAvg8Trainer',
            'fedavg8b': 'FedAvg8bTrainer',
            'fedavg9': 'FedAvg9Trainer',
            'fedavg10': 'FedAvg10Trainer',
            'fedavg10b': 'FedAvg10bTrainer',
            'fedavg11': 'FedAvg11Trainer',
            'fedavg12': 'FedAvg12Trainer',
            'fedavg13': 'FedAvg13Trainer', 
            'fedavg14': 'FedAvg14Trainer',
            'fedavg15': 'FedAvg15Trainer', # TEST
            'game1': 'Game1Trainer',
            'game2': 'Game2Trainer',
            'pretrain1':'Pretrain1Trainer',
            'Unigame':'UnigameTrainer',
            'Rightgame':'RightgameTrainer',
            'Opgame':'OpgameTrainer',
            }
OPTIMIZERS = TRAINERS.keys()


class ModelConfig(object):
    def __init__(self):
        pass

    def __call__(self, dataset, model):
        dataset = dataset.split('_')[0]

        if dataset == 'mnist' or dataset == 'nist':
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 10}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 10}
        elif dataset == 'emnist':
            if model == 'logistic':
                return {'input_shape': 784, 'num_class': 26}
            else:
                return {'input_shape':(1,28,28), 'num_class':26}
        elif dataset == 'femnist':
            if model == 'logistic':
                return {'input_shape': 784, 'num_class': 62}
            else:
                return {'input_shape':(1,28,28), 'num_class':52}
        elif dataset == 'cifar10':
            return {'input_shape': (3, 32, 32), 'num_class': 10}
        elif dataset == 'sent140':
            sent140 = {'bag_dnn': {'num_class': 2},
                       'stacked_lstm': {'seq_len': 25, 'num_class': 2, 'num_hidden': 100},
                       'stacked_lstm_no_embeddings': {'seq_len': 25, 'num_class': 2, 'num_hidden': 100}
                       }
            return sent140[model]
        elif dataset == 'shakespeare':
            shakespeare = {'stacked_lstm': {'seq_len': 80, 'emb_dim': 80, 'num_hidden': 256}
                           }
            return shakespeare[model]
        elif dataset == 'synthetic':
            return {'input_shape': 60, 'num_class': 10}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))


MODEL_PARAMS = ModelConfig()