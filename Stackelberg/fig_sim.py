'''
    *this program is used for drawing fig.1 
    *for k=1~N, calculate simulated and real expectation of time, 
    *need 8 pictures
        * N = 100 / 40
        * qi = pk, 1/n
        * computation time dominated, similar time

'''

import os 
import json

import numpy as np
import matlab.engine
eng= matlab.engine.start_matlab()

import matplotlib.pyplot as plt 

np.random.seed(7)

def convert_dict_2_json(dictContent, file_name, file_path='cache/'):
    file = os.path.join(file_path, file_name)
    with open(file, 'w') as fp:
        json.dump(dictContent, fp)

def convert_json_2_dict(file_name, file_path='cache/'):
    file = os.path.join(file_path, file_name)
    with open(file) as fp:
        dictContent = json.load(fp)

    return dictContent

def draw_fig(N, Q, T, Tau, R, fileName):
    '''
        - N number of clients
        - Q sampling prob
        - T t_i
        - Tau tau_i
        - R round
    '''
    points = {'simu':[], 'real':[]}

    ## calc simulated t
    

    for k in range(1, 1+N):
        ## tSim        
        points['simu'].append(np.sum(Q * (k*T+Tau)))

        ## tReal
        tReal = 0
        for i in range(R):
            selClients = np.random.choice(N, k, p=Q)
            T_ = T[selClients] + [0]*(N-k)
            Tau_ = Tau[selClients] + [0]*(N-k)
            if N == 100:
                Tr = eng.m_solve_root_n100(T_, Tau_, 1)
            elif N == 40:
                Tr = eng.m_solve_root_n40(T_, Tau_, 1)
            tReal += Tr 

        points['real'].append(tReal/R)
    
    convert_dict_2_json(points, fileName, './fig1')

    plt.plot(range(1,1+N), points['simu'])
    plt.plot(range(1,1+N), points['real'])

    print("%s DONE."%fileName)


## Setting 1
##  - Uniform time 
##  - computation time dominate

## - N 100
TComm = np.random.uniform(0.005,0.05,100)
TComp = np.random.uniform(0.1,1,100)
draw_fig(100, [1/100]*100, TComm.tolist(), TComp.tolist(), 10000, 'test.json')

## - N 40


plt.show()
