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

class BaseTrainer(object):
    def __init__(self, options, dataset, model=None, optimizer=None, name='', worker=None):
        if model is not None and optimizer is not None:
            self.worker = Worker(model, optimizer, options)
        elif worker is not None:
            self.worker = worker
        else:
            raise ValueError("Unable to establish a worker! Check your input parameter!")
        print('>>> Activate a worker for training')
        
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.isSysHeter = options['sys_heter']

        ##  grad_path -> path of gk
        # self.grad_path = options['grad_path']
        # self.grad_clients = convert_json_2_dict(self.grad_path, '.')

        self.without_r = options['without_r'] # with replacement 
        self.update_fre = options['update_fre']
        
        self.real_round = 100
        self.all_train_data_num = 0
        self.alpha = 1.0
        self.clients = self.setup_clients(dataset)
        # self.get_num_label()
        self.selected_clients = None 
        # self.client_times = np.zeros(self.numOfClients)
        
        self.decay = 'round'
        if 'mnist' in options['dataset']:
            # decay
            self.decay = options['decay']
            
        
        assert len(self.clients) > 0
        # print('>>> Initialize {} clients in total'.format(len(self.clients)))
        # print("data distribution", [len(c.train_data) for c in self.clients])

        self.numOfClients = len(self.clients)
        self.client_times = np.zeros(self.numOfClients)
        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_every = options['eval_every']
        self.simple_average = not options['noaverage']
        # print('>>> Weigh updates by {}'.format(
        #     'simple average' if self.simple_average else 'sample numbers'))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()
        # might be update the params use the last time params

        
        if self.isSysHeter:
            self.c0 = options['c0']
            self.time_flag = options['time_flag']
            # self.optim_method = options['optim_method']
            self.optim_method = 'matlab'
            self.client_times = options['client_times']
            if options['algo'] in ['game1','game2','Unigame','Opgame','Rightgame','fedavg6','fedavg8','fedavg10','fedavg6b','fedavg8b','fedavg10b','fedavg7']:
                import matlab.engine
                self.eng= matlab.engine.start_matlab()
        
        # self.client_times = options['client_times']

        # algo name
        algo_name = algo = options['algo']

        if  algo in ['fedavg8','fedavg8b']:
            algo_name += 't%s' % options['update_fre']

        if self.without_r:
            algo_name = 'w' + algo_name
        
        if self.decay != 'round':
            algo_name = 's' + algo_name

        if self.isSysHeter:
            algo_name += "C{}.".format(self.c0)

        test_n = options['test_num']
        dataset = options['dataset']
        E = options['num_epoch']
        K = options['clients_per_round']
        R = options['num_round']
        SEED = options['seed']
        self.smooth_delta = 1
        # Info used 
        if self.isSysHeter:
            if  options['algo'] in ['fedavg8','fedavg8b']:
                self.name = "test{}_{}_{}{}_e{}_k{}_round{}_seed{}_time{}{}.json".format(options['test_num'],options['dataset'], algo_name, options['update_fre'],options['num_epoch'], options['clients_per_round'], options['num_round'], options['seed'],options['time_seed'],options['time_flag'])
            else:    
                if self.smooth_delta == 1:
                    self.name = "test{}_{}_{}_e{}_k{}_round{}_seed{}_time{}{}.json".format(options['test_num'],options['dataset'], algo_name, options['num_epoch'], options['clients_per_round'], options['num_round'], options['seed'],options['time_seed'],options['time_flag'])
                else:
                    self.name = "test{}_{}_{}_e{}_k{}_round{}_seed{}_time{}{}_smt{}_{}.json".format(options['test_num'],options['dataset'], algo_name, options['num_epoch'], options['clients_per_round'], options['num_round'], options['seed'],options['time_seed'],options['time_flag'],self.smooth_thd,self.smooth_delta)

                    self.name = "test{}_{}_{}T{}D{}_e{}_k{}_round{}_seed{}_time{}{}.json".format(options['test_num'],options['dataset'], algo_name,self.smooth_thd,self.smooth_delta,options['num_epoch'], options['clients_per_round'], options['num_round'], options['seed'],options['time_seed'],options['time_flag'])
        else:
            # STA heter
            self.name='T%s.D%s.ALG%s.E%s.K%s.R%s.S%d.G%s.json' % (test_n, dataset, algo_name, E, K, R, SEED, self.grad_path.split('.')[-1])

        if options['model'] == 'cnn':
            self.name += '.Mcnn' # Model Lenet
        elif options['model'] == 'lenet':
            self.name += '.Mlenet' #

        if options['lr'] != 0.1:
            self.name += '.LR%s' % options['lr']
        
        cpath = dir_path = os.path.dirname(os.path.realpath(__file__))

        self.path = os.path.join(cpath, '../../test-result', options['log'])

        self.loss = 0

        num_sample_clients = [len(c.train_data) for c in self.clients]
        pk_clients = np.array(num_sample_clients) / np.sum(num_sample_clients)

            
        if self.isSysHeter:
            options.pop('client_times')
        self.logDict = {
            'experiment_time':datetime.now().strftime("%D:%H:%M:%S"),
            'info': options,
            'num_sample_clients':num_sample_clients,
            'client_times':self.client_times.tolist(),
            'budget': 0,
            'alpha': 1,
            'pk':pk_clients.tolist(),
            'real_sample':[],
            'num_label':[],
            'qk':[],
            'Pk':[],
            'ck':[],
            'vk':[],
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
            c.pk = self.logDict['pk'][i]
        ## Temporary print
        # print('time', self.logDict['client_times'])
        # print('data', self.logDict['num_sample_clients'])





        for i, c in enumerate(self.clients):
            assert i == c.cid


    def select_clients_with_prob_without_replacement(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        index = np.random.choice(len(self.clients), num_clients,replace=False, p=self.prob)
        index = sorted(index.tolist())

        select_clients = []
        select_index = []
        repeated_times = []
        for i in index:
            if i not in select_index:
                select_clients.append(self.clients[i])
                select_index.append(i)
                repeated_times.append(1)
            else:
                repeated_times[-1] += 1
        # check
        print("check",select_index == [c.cid for c in select_clients])
        return select_clients, repeated_times


    def get_qk_heter(self, A, B, qk_prev, divc):
        def func(x):
            # print("->",np.sum(A*x) * (np.sum(B/x) + self.c0) / divc)
            return np.sum(A*x) * (np.sum(B/x) + self.c0) / divc
         
        def con_p(x):
            return np.sum(x) - 1
        
        def con_larger_0(x):
            return x - 1e-7
        
        def con_less_1(x):
            return 1 - x

        cons = (
            {'type':'eq', 'fun':con_p},
            {'type':'ineq', 'fun':con_larger_0},
            {'type':'ineq', 'fun':con_less_1}
        )
        x0 = qk_prev
        res = minimize(func, x0,constraints=cons,options={"disp":True,})
        print(res.message)


        return res.success, res.x

    def get_qk(self, A, B, qk_prev):
        divc = 1
        ret = None
        while True:
            try:
                flg, ret = self.get_qk_heter(A,B,qk_prev,divc)
                if flg == True:
                    print("get qk", flg, ret, divc)
                    return ret
            except:
                pass 
            divc *= 10
            print("get qk Wrong", divc)
    # A0/B0 = 0
    def get_U(self, probs,A,R):
        ret = 0.0
        for i in range(20):
            ret += (1-probs[0][i])*A[0][i]/probs[0][i]
            # print(ret,A[0][i], probs[0][i])
            
        return ret/R

    def get_qk_matlab2(self, Ai, Bi, Ci, Mnn, Mmx):
        import matlab
        LOWER_BOUND = 0.01
        UPPER_BOUND = 1-1e-7
        N = 20
        R = 1000*1.0
        budget = 3.0
        A = matlab.double([x for x in Ai])
        B = matlab.double([x for x in Bi])
        C = matlab.double([x for x in Ci])
        lb = matlab.double([LOWER_BOUND]*N)
        ub = matlab.double([UPPER_BOUND]*N)
        x0 = matlab.double([1/N]*N)
        stp = 0.1
        M = 1.0*Mnn
        ret = self.eng.M_duipai_20(x0,A,B,C,lb,ub, R, budget, Mnn)
        ans = self.get_U(ret,A,R)
        while M <= Mmx:
            A = matlab.double([x for x in Ai])
            B = matlab.double([x for x in Bi])
            C = matlab.double([x for x in Ci])
            lb = matlab.double([LOWER_BOUND]*N)
            ub = matlab.double([UPPER_BOUND]*N)
            x0 = matlab.double([1/N]*N)
            now = self.eng.M_duipai_20(x0,A,B,C,lb,ub, R, budget,M)
            
            tmp = self.get_U(now,A,R)
            if tmp < ans and tmp > 0:
                ans = tmp
                ret = now
            M += stp
            print(M, tmp, np.array(ret).squeeze())

        return np.array(ret).squeeze()

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
        

    def calc_time(self, selected_clients):
        times = [self.client_times[c.cid] for c in selected_clients]
        if len(times) == 0:
            return 0.0
        return np.max(times).item()

    def save_log(self, ):
        # print()
        convert_dict_2_json(self.logDict, self.name, self.path)
        # save_object(self.Log, self.log_file_name)

    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        tmp = 0
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            # print(test_data)
            # raise Exception("Pause")
            c = Client(tmp, group, train_data[user], test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
            tmp += 1
        return all_clients

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
        self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.clients:
            # c.set_flat_model_params(self.latest_model)
            # print("check client", c.worker == self.worker )
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
