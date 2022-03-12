

import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.file_utils import convert_json_2_dict

# SEED_LIST = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21]
SEED_LIST = [1,]
DATASET = 'mnist_niid1_7_0'
E=50
K=3
ROUND=200
LOG='6-25'

ALGOS=['lx_s1','lx_s2t','bing_s1','bing_s2300','bing_s250','bing_s210']

for algo in ALGOS:
    loss = []

    for seed in SEED_LIST:
        name = 'test1_{}_{}_e{}_k{}_round{}_seed{}.json'.format(DATASET,algo,E,K,ROUND,seed)
        path = os.path.join('log', LOG)

        dict = convert_json_2_dict(name, path)
        loss.append(dict['global_loss'])

    loss = np.mean(loss, axis=0)

    plt.plot(loss)

plt.legend(ALGOS, loc='upper right')
plt.show()
    
