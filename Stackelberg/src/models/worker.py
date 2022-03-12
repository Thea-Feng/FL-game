from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch
import numpy as np

criterion = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """
    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False

        # Setup local model and evaluate its statics
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])

    @property
    def model_bits(self):
        return self.model_bytes * 8

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        # 可以假设为logistic train
        train_loss = train_acc = train_total = 0
        for epoch in range(self.num_epoch):
            train_loss = train_acc = train_total = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):
                # from IPython import embed
                # embed()
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                if torch.isnan(pred.max()):
                    from IPython import embed
                    embed()

                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss/train_total,
                       "acc": train_acc/train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        # model.eval ()的作用是 不启用 Batch Normalization 和 Dropout
        
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:
                # print("test")
                # from IPython import embed
                # embed()
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss

    def calc_grad(self, train_dataloader, **kwargs):
        flat_grad = self.get_flat_grads(train_dataloader)
        flat_grad_norm = np.linalg.norm(flat_grad.cpu().detach().numpy())
        # print("method 1 | average flat grad norm", flat_grad_norm)
        return flat_grad_norm

    def calc_grad_sample_one(self, dataloader, **kwargs):
        # Calc gradient of each data
        # batchsize of dataloader should be one

        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        flatGradNormOfPoints = []
        for x, y in dataloader:
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)

            lossOnePoint = criterion(pred, y) * y.size(0)
            flatGradsOnePoint = get_flat_grad(lossOnePoint, self.model.parameters(), create_graph=True)
            flatGradNormOnePoint = np.linalg.norm(flatGradsOnePoint.cpu().detach().numpy())
            flatGradNormOfPoints.append(flatGradNormOnePoint)

            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        flat_grad_norm = np.linalg.norm(flat_grads.cpu().detach().numpy())

        print("flat grad norm of points", flatGradNormOfPoints)
        print("average flat grad norm", flat_grad_norm)

        return flat_grad_norm

    def calc_grad_sample_all(self, dataloader, **kwargs):
        # Calc gradient of all
        # batchsize of dataloader should be datasize

        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        flatGradNormOfPoints = []
        for x, y in dataloader:
            print("->")
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        flat_grad_norm = np.linalg.norm(flat_grads.cpu().detach().numpy())
        print("method 3 | average flat grad norm", flat_grad_norm)

        return flat_grad_norm
        
class LrdWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrdWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        # for i in range(self.num_epoch*10):

        # get flat gradient norm
        flat_grad_norm_epochs = []

        for i in range(self.num_epoch):
            x, y = next(iter(train_dataloader))
            
            if self.gpu:
                x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss = criterion(pred, y)
            # loss_ = criterion(pred, y)
            # Calc flag grad norm
            # flat_grad = get_flat_grad(loss_, self.model.parameters(), create_graph=True)
            # flat_grad_norm = np.linalg.norm(flat_grad.cpu().detach().numpy())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            
            self.optimizer.step()
            
            # flat_grad_norm = np.linalg.norm(flat_grad.cpu().detach().numpy())
            # flat_grad_norm_epochs.append(flat_grad_norm)
            
            # print("flat Grad 1", flat_grad_norm)
            # print("Flat Grad 2", flat_grad_norm_)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size

        # calc flat gradient norm mean
        # flat_grad_norm_mean = np.mean(flat_grad_norm_epochs) 
        # print("flat grad norm", flat_grad_norm_epochs)
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {
                "comp": comp,
                "loss": train_loss/train_total,
                "acc": train_acc/train_total,
                # 'gk': flat_grad_norm_mean,
            }
        return_dict.update(param_dict)
        return local_solution, return_dict

    # def calc_grad(self, train_dataloader, **kwargs):
    #     flat_grad = self.get_flat_grads(train_dataloader)
    #     flat_grad_norm = np.linalg.norm(flat_grad.cpu().detach().numpy())
    #     return flat_grad_norm


class LrAdjustWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrAdjustWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, **kwargs):
        m = kwargs['multiplier']
        current_lr = self.optimizer.get_current_lr()
        self.optimizer.set_lr(current_lr * m)
        
        self.model.train()
        train_loss = train_acc = train_total = 0
        # for i in range(self.num_epoch*10):
        for i in range(self.num_epoch):
            x, y = next(iter(train_dataloader))
            
            if self.gpu:
                x, y = x.cuda(), y.cuda()
        
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()
            
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)
            
            train_loss += loss.item() * y.size(0)
            train_acc += correct
        train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
            "loss": train_loss/train_total,
                "acc": train_acc/train_total}
        return_dict.update(param_dict)
        
        self.optimizer.set_lr(current_lr)
        return local_solution, return_dict