import numpy as np

# generate time

datasize = [236, 56, 512, 69, 51, 161, 931, 57, 46, 125, 54, 77, 88, 84, 147, 191, 94, 69, 52, 192, 46, 47, 87, 382, 236, 192, 125, 46, 361, 57, 90, 213, 58, 76, 79, 136, 73, 252, 53, 442, 1622, 52, 102, 808, 146, 60, 2012, 322, 61, 47, 423, 92, 243, 116, 93, 92, 559, 46, 282, 72, 82, 51, 47, 55, 55, 50, 234, 60, 65, 95, 51, 107, 650, 107, 235, 98, 114, 176, 199, 81, 1134, 61, 112, 49, 78, 118, 128, 68, 606, 96, 54, 144, 339, 253, 330, 59, 61, 63, 372, 179]


def generate_time(num_clients, SEED, flg='random'):
    x = flg.find('_')
    if x != -1:
        mode = flg[:x]
    else:
        mode = flg 
        
    dist = flg[x+1:]
    if mode == 'random':
        return generate_random_time(num_clients, SEED, dist)
    elif mode == 'constant':
        return generate_constant_time(num_clients,)
    elif mode == 'posCas':
        return generate_pos_cascade_time(num_clients, SEED)
    elif mode == 'pos':
        return generate_pos_time(num_clients, SEED,dist)


def generate_random_time(num_clients, SEED, dist='exp'):
    np.random.seed(10 + SEED)
    if dist == 'exp':
        times = np.random.exponential(1, num_clients)
        # for i in range(num_clients):
        #     if times[i] == np.max(times):
        #         times[i] += 10
        #         break 
    elif dist == 'expCas':  
        print("Exponential Dist")
        times = np.random.exponential(1, num_clients+5)
        times.sort()
        times = times[4:num_clients+4]
        np.random.shuffle(times)
    elif dist == 'uniform':
        TL = 1
        TR = 19 
        times = np.random.uniform(TL,TR,num_clients)
    elif dist == 'normal':
        times = np.random.normal(loc=1,scale=0.34,size=num_clients)
        times = np.abs(times)

    # times.sort()
    # clients = [(data,i) for i,data in enumerate(datasize) ]
    # clients.sort()
    # new_times = [None]*num_clients
    # for i, x in enumerate(clients):
    #     new_times[x[1]] = times[i]
    return times
    
def generate_constant_time(num_clients):
    return np.array([1 for i in range(num_clients)])

def generate_pos_cascade_time(num_clients, SEED):
    np.random.seed(10 + SEED)
    # num_samples = np.random.lognormal(5.5, 1, (num_clients+30))
    times = np.random.exponential(1, num_clients+30)
    times.sort()
    times = times[15:num_clients+15]
    clients = [(data,i) for i,data in enumerate(datasize) ]
    clients.sort()
    new_times = [None]*num_clients
    for i, x in enumerate(clients):
        new_times[x[1]] = times[i]
    return np.array(new_times)

def generate_pos_time(num_clients, SEED, dist='exp'):


    np.random.seed(10 + SEED)

    if dist == 'exp':  
        print("Exponential Dist")
        times = np.random.exponential(1, num_clients)
    elif dist == 'uniform':
        times = np.random.uniform(0.01,1.99,num_clients)
    elif dist == 'normal':
        times = np.random.normal(loc=1,scale=0.34,size=100)
        times = np.abs(times)

    times.sort()
    clients = [(data,i) for i,data in enumerate(datasize) ]
    clients.sort()
    new_times = [None]*num_clients
    for i, x in enumerate(clients):
        new_times[x[1]] = times[i]
    return np.array(new_times)

def generate_real_time(num_clients, **kwargs):
    '''
        generate real time
        
        * num_clients: number of clients
        * SEED 
        * is_software: this is for software experiment

    '''


    np.random.seed(5 + kwargs['SEED'])

    ftot = 1.0


    # t_comp_mean = t_comm_mean * kwargs['frac_comp_comm']
    # comm_times = np.random.uniform(0.005,0.05,num_clients)
    # comp_times = np.random.uniform(0.1,1,num_clients)
    # comm_times = np.random.exponential(t_comm_mean, num_clients)
    # comp_times = np.random.exponential(t_comp_mean, num_clients)

    # ## 
    # comm_times = np.random.exponential(1, num_clients)
    # comp_times = np.random.exponential(1, num_clients)

    # real_times = comm_times*kwargs['k']/ftot + comp_times
    # print(comm_times.tolist())
    # print(comp_times.tolist())
    # return real_times 

    ## Testing, time possitive correlated
    comm_times = np.random.exponential(1, num_clients)
    comp_times = np.random.exponential(1, num_clients)

    ## Generate real times
    ## ftot is contained in t_i
    realTimes = comm_times * kwargs['k'] + comp_times 

    cliDataSize = np.array([236, 56, 512, 69, 51, 161, 931, 57, 46, 125, 54, 77, 88, 84, 147, 191, 94, 69, 52, 192, 46, 47, 87, 382, 236, 192, 125, 46, 361, 57, 90, 213, 58, 76, 79, 136, 73, 252, 53, 442, 1622, 52, 102, 808, 146, 60, 2012, 322, 61, 47, 423, 92, 243, 116, 93, 92, 559, 46, 282, 72, 82, 51, 47, 55, 55, 50, 234, 60, 65, 95, 51, 107, 650, 107, 235, 98, 114, 176, 199, 81, 1134, 61, 112, 49, 78, 118, 128, 68, 606, 96, 54, 144, 339, 253, 330, 59, 61, 63, 372, 179])

    dataPair = [(d,i) for i,d in enumerate(cliDataSize)]
    realTimes.sort()
    dataPair.sort()

    time_ = [0] * num_clients
    for i, e in enumerate(dataPair):
        time_[e[1]] = realTimes[i]

    print("time", time_)
    return np.array(time_)

def generate_real_time_v1(num_clients, **kwargs):
    '''
        Created for Journal Experiment 1

        generate time
        
        * num_clients: number of clients
        * SEED 
        * is_software: this is for software experiment

    '''

    np.random.seed(5 + kwargs['SEED'])

    ftot = 1.0

    if kwargs['is_software']:
        ### Software time setting
        ### - t ~ exp(1)
        ### - tau ~ exp(1)


        ## Communication time
        t_comm_clients = np.random.exponential(1,num_clients)
        ## Computation time
        t_comp_clients = np.random.exponential(1,num_clients)

    else:
        ###    Hardware time setting
        ###    - t ~ u(0.2,5)
        ###    - tau = 5


        ## Communication time
        t_comm_clients = np.random.uniform(0.2,5,num_clients)
        t_comp_clients = np.ones(num_clients) * 5
    
  
    real_times = t_comm_clients*kwargs['k']/ftot + t_comp_clients

    return real_times 

 