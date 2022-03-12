# utils used for logging
import pickle

class RunTimeInfo:
    def __init__(self, ):
        # self.round = []
        self.loss = []
        self.q_k = []
        self.G_k = []
        self.p_k = []
        self.times = []
        self.data_dist = []     # data distribution
        self.time_dist = []     # time distribution

    def add_distribution(self, data, time):
        self.data_dist = data 
        self.time_dist = time 

    def add_info(self, loss, q_k, G_k, p_k, t_i):
        self.loss.append(loss)
        self.q_k.append(q_k)
        self.G_k.append(G_k)
        self.p_k.append(p_k)
        self.times.append(t_i)
        

def save_object(obj, file_name):
    with open(file_name, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(file_name):
    with open(file_name, 'rb') as input:
        obj = pickle.load(input)
    return obj
