import time
from torch.utils.data import DataLoader


class Client(object):
    """Base class for all local clients

    Outputs of gradients or local_solutions will be converted to np.array
    in order to save CUDA memory.
    """
    def __init__(self, cid, group, train_data, test_data, batch_size, worker):
        self.cid = cid
        self.group = group
        self.worker = worker

        self.train_data = train_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # new added 
        self.pk = None # nk / n
        self.qk = None # qk
        self.Gk = None
        self.ck = None
        self.vk = None 
        self.real = None
        self.round = None

    def get_model_params(self):
        """Get model parameters"""
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def get_flat_grads(self):
        """Get model gradient"""
        grad_in_tenser = self.worker.get_flat_grads(self.train_dataloader)
        return grad_in_tenser.cpu().detach().numpy()

    def solve_grad(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}

        grads = self.get_flat_grads()  # Return grad in numpy array

        return (len(self.train_data), grads), stats

    # new added
    def get_grad(self, **kwargs):
        trainDataLoader = DataLoader(self.train_data, batch_size=len(self.train_data), shuffle=False)
        worker_grad = self.worker.calc_grad(trainDataLoader, **kwargs)
        # here we get the norm of the gradient

        # worker_grad = self.worker.calc_grad(self.train_dataloader, **kwargs)
        # print("Client {}".format(self.cid) ,end=" ")
        # for x, y in self.train_dataloader:
        #     print("datasize is {}".format(y.size(0)))
        return worker_grad

    def get_grad_sample_one(self, **kwargs):
        print("client {}".format(self.cid))
        trainDataLoader = DataLoader(self.train_data, batch_size=1, shuffle=False)
        self.worker.calc_grad_sample_one(trainDataLoader, **kwargs)

    def get_grad_smaple_all(self, **kwargs):
        print("data size", len(self.train_data))
        trainDataLoader = DataLoader(self.train_data, batch_size=len(self.train_data), shuffle=False)
        self.worker.calc_grad_sample_all(trainDataLoader, **kwargs)

    def local_train(self, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats

    def local_test(self, use_eval_data=True):
        """Test current model on local eval data

        Returns:
            1. tot_correct: total # correct predictions
            2. test_samples: int
        """
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        tot_correct, loss = self.worker.local_test(dataloader)

        return tot_correct, len(dataset), loss
