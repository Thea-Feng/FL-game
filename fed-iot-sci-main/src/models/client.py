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
        # data type ? can this be transfered by json ?
        self.test_data = test_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # new added 
        self.pk = None # nk / n
        self.qk = None # qk
        self.Gk = 0 

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
        return worker_grad


    def local_train(self, args, **kwargs):
        
        lr = args['lr']
        self.worker.optimizer.set_lr(lr)
        latest_model = args['latest_model_tensor'] # model should be tensor
        comp_lim = args['comp_lim']
        tStart = time.time()
        
        for _ in range(comp_lim):
            self.set_flat_model_params(latest_model)
            local_solution, worker_stats = self.worker.local_train(self.train_dataloader, **kwargs)
        tComp = time.time() - tStart 

        return {
            'data_size': len(self.train_data),
            'local_solution_list': local_solution.tolist(),
            't_comp': tComp,
            'local_loss': worker_stats['loss'],
            'local_acc': worker_stats['acc']
        }

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
