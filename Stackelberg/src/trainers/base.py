import numpy as np
import torch
import time
from src.models.client import Client
from src.utils.worker_utils import Metrics
from src.models.worker import Worker
from src.utils.file_utils import convert_dict_2_json, convert_json_2_dict
import os
from scipy.optimize import minimize
from datetime import datetime 
# import matlab
# import matlab.engine

from torch import Tensor
from src.models.client import Client
import src.communication.comm_config as connConfig


class BaseTrainer(object):
    def __init__(self, options, trainerConfig):

        self.worker = trainerConfig['worker']

        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.is_sys_heter = options['is_sys_heter']
        self.all_train_data_num = trainerConfig['all_train_data_num']
        self.clients = trainerConfig['clients']
        self.socketServer = trainerConfig['socketServer']
        self.debug = trainerConfig['debug']
        self.numOfClients = options['num_clients']
        self.without_rp = options['without_rp']
        self.decay = options['decay']

        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.eval_every = options['eval_every']
        self.simple_average = not options['noaverage']
        if 'game1'or 'right1' or 'unifrom' in options['algo']:
            import matlab
            self.eng= matlab.engine.start_matlab()

        self.latest_model = self.worker.get_flat_model_params().detach()
        
        # Add time heterogeneity
        self.client_times = trainerConfig['client_times']
        self.update_fre = options['update_fre']
        # Info used 
        '''
            * T test
            * DS dataset
            * AG algorithm t-transform
            * E, K
            * R round
            * SD seed
            * TS time seed
        '''
           
        self.file_name = "T{}.DS{}.AG{}.E{}.R{}.SD{}.TS{}.json".format(options['test_num'], options['num_round'], options['seed'],options['time_seed'])

        cpath = dir_path = os.path.dirname(os.path.realpath(__file__))

        self.file_path = os.path.join(cpath, '../..','log', options['experiment_folder'])

        num_sample_clients = [len(c.train_data) for c in self.clients]
        pk_clients = np.array(num_sample_clients) / np.sum(num_sample_clients)
        
        for i, c in enumerate(self.clients):
            c.pk = pk_clients[i]
            c.ck = options['C'][i]
            c.vk = options['v'][i]
        
            
        if self.isSysHeter:
            options.pop('client_times')
        self.logDict = {
            'experiment_time':datetime.now().strftime("%D:%H:%M:%S"),
            'info': options,
            'num_sample_clients':num_sample_clients,
            'client_times':self.client_times.tolist(),
            'budget': options['budget'],
            'alpha': options['alpha'],
            'pk':pk_clients.tolist(),
            'real_sample':[],
            'num_label':[],
            'qk':[],
            'Pk':[],
            'ck':[c.ck for c in self.clients],
            'vk':[c.vk for c in self.clients],
            'gk':[],
            'pG':[],
            'sel_clients':[],
            'lambda':[],
            'tk':[],
            'loss':[],
            'acc':[],
            'time':[],
            'sample_diff':[],
            'real_gk':[],
            
        }
        

        for i, c in enumerate(self.clients):
            assert i == c.cid


   

    # get qk using matlab
    def get_qk_matlab(self, A, B, C, budget,alpha):
        import matlab
        LOWER_BOUND = 0.01
        UPPER_BOUND = 1-1e-7
        N = self.numOfClients
        R = self.real_round*1.0


        A = matlab.double([x for x in A])
        B = matlab.double([x for x in B])
        C = matlab.double([x for x in C])
        lb = matlab.double([LOWER_BOUND]*N)
        ub = matlab.double([UPPER_BOUND]*N)
        x0 = matlab.double([1/N]*N)


        if N == 100:
            if np.array(self.logDict['vk']).sum() == 0.0:
                ret = self.eng.M_fmincon_100v0(x0,A,B,C,lb,ub, R, budget,alpha)
            else:
                ret = self.eng.M_fmincon_100(x0,A,B,C,lb,ub, R, budget,alpha)
        elif N == 40:
            ret = self.eng.M_fmincon_40(x0,A,B,C,lb,ub,R,budget,alpha)
        elif N == 20:
            ret = self.eng.M_fmincon_20(x0,A,B,C,lb,ub,R,budget,alpha)
        return np.array(ret).squeeze()

    def get_qk_c0(slef, A, B):
        sqrt_b0_over_a0 = np.sqrt(B/A)
        return sqrt_b0_over_a0 / np.sum(sqrt_b0_over_a0)
        
    def get_real_time(self, stats):
        tcommOfClients = [ stat['tcomm'] for stat in stats ]
        return max(tcommOfClients) 


    def save_log(self, ):
        convert_dict_2_json(self.logExperimentInfo, self.file_name, self.file_path)

  
    def end_train(self, ):
        for i, c in enumerate(self.clients):
            info = self.socketServer.send_msg(
                message={
                    'type': connConfig.MSG_END_EXPERIMENT,
                    'msg': None,
                    'tag': '',
                },
                reciver={
                    'type': connConfig.CLIENT,
                    'num': c.cid,
                })
            if self.debug:
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))


    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

   
    def train(self):
        raise NotImplementedError

    def select_clients(self, seed=1):
        """Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            1. seed: random seed
            2. num_clients: number of clients to select; default 20
                note that within function, num_clients is set to min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        """
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def local_train(self, round_i, selected_clients):
        

        solns = []  # client solutions
        stats = []  # Client comp and comm costs

        lr = self.worker.optimizer.get_current_lr()

        # Step one: send selection signal & latest model 
        for i, c in enumerate(selected_clients):
            info = self.socketServer.send_msg(
                message={
                    'type': connConfig.MSG_LOCAL_TRAIN,
                    'msg': {
                        'latest_model_list': self.latest_model.tolist(),
                        'lr': lr,
                        'round_i': round_i,
                    },
                    'tag': '',
                },
                reciver={
                    'type': connConfig.CLIENT,
                    'num': c.cid,
                })
            if self.debug:
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
            
        # Step two: Recieve local model & time 
        for i, c in enumerate(selected_clients):
            self.socketServer.send_signal(connConfig.FIRST_SHAKE_HANDS,{'type':connConfig.CLIENT,'num':c.cid })
            message, info = self.socketServer.recv_msg({'type':connConfig.CLIENT,'num':c.cid })
            if self.debug:
                print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))

            msg = message['msg']
            soln = (msg['data_size'], Tensor(msg['local_solution_list']))
            stat = {
                'client': c.cid,
                'tcomp': msg['t_comp'],
                'tcomm': info['time'],
                'vcomm': info['size']/info['time'],
                'loss': msg['local_loss'],
                'acc': msg['local_acc']
            }

            print(">>>>>>>> Round {} | CID: {} loss: {:.5f}, acc: {:.5f}, t_comm: {:.5f}, t_comp: {:.5f}".format(round_i, c.cid, stat['loss'], stat['acc'], stat['tcomm'], stat['tcomp']))

            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def get_grad(self, c):
        info = self.socketServer.send_msg(
            message={
                'type': connConfig.MSG_GET_GRAD,
                'msg': {
                    'latest_model_list': self.latest_model.tolist()
                },
                'tag': ''
            }, 
            reciver={
                'type': connConfig.CLIENT,
                'num': c.cid
            }
        )
        if self.debug:
            print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))

        message, info = self.socketServer.recv_msg(         
            sender={'type': connConfig.CLIENT,'num': c.cid}
        )

        msg = message['msg']

        if self.debug:
            print("size {}, time {}, speed {} byte/s".format(info['size'], info['time'], info['size']/info['time']))
        
        return msg['c_grad']
   

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def select_clients(self, seed=1):
        """Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            1. seed: random seed
            2. num_clients: number of clients to select; default 20
                note that within function, num_clients is set to min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        """
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train()
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def aggregate(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        if self.simple_average:
            num = 0
            for num_sample, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            for num_sample, local_solution in solns:
                averaged_solution += num_sample * local_solution
            averaged_solution /= self.all_train_data_num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

    def test_latest_model_on_traindata(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # Record the global gradient
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # Measure the gradient difference
        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                   stats_from_train_data['gradnorm'], difference, end_time-begin_time))
            print('=' * 102 + "\n")

        self.loss = stats_from_train_data['loss']
        return global_grads

    def test_latest_model_on_evaldata(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)
        self.acc = stats_from_eval_data['acc']

    def local_test(self, use_eval_data=True):
        assert self.latest_model is not None
        # self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.clients:
            c.set_flat_model_params(self.latest_model)
            tot_correct, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.cid for c in self.clients]
        groups = [c.group for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids, 'groups': groups}

        return stats

    def get_num_label(self, ):
        num_label = []
        for c in self.clients:
            data = c.train_data 
            labels = [ x[1] for x in data ]
            # labels = data['y']
            # print("datasize %d, num label %d" % (len(data), len(set(labels))))
            self.logDict['num_label'].append(len(set(labels)))
            num_label.append(len(set(labels)))
        # print(num_label)
        # raise Exception("Pause")
