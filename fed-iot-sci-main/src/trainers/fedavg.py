from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD

from src.trainers.base import BaseTrainer
from src.models.model import choose_model
# from src.models.worker import LrdWorker
from src.optimizers.gd import GD
# import numpy as np
import torch
# from scipy.optimize import minimize
# Scheme 5
# Sample all clients
# Average

class FedAvgTrainer(BaseTrainer):
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(FedAvgTrainer, self).__init__(options, dataset, model, self.optimizer)
        self.prob = None 
        self.log_q_k = self.prob 

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Add log to log class
            self.Log.add_info(self.log_round_loss, self.log_q_k, self.log_G_k, self.log_p_k, self.log_round_time)


            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)


            # calc and log round time 
            time_round_i = self.calc_time(selected_clients)
            self.log_round_time = time_round_i

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

            # update log info
            self.log_G_k = self.update_log_G_k()

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)

        num = 0
        for num_sample, local_solution in solns:
            num += 1
            averaged_solution += local_solution
        averaged_solution /= num

        # if self.simple_average:
        #     num = 0
        #     for num_sample, local_solution in solns:
        #         num += 1
        #         averaged_solution += local_solution
        #     averaged_solution /= num
        # else:
        #     for num_sample, local_solution in solns:
        #         averaged_solution += num_sample * local_solution
        #     averaged_solution /= self.all_train_data_num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()