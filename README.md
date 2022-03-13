# Incentive Mechanism Design for Federated Learning with Partial Client Participation
# Part 1 Hardware prototype: Fed-IoT-Sci
# Under fed-iot-sci-main folder
**This code is used for randomly selecting clients for Federated Learning IoT experiment**

Our code is based on the code for [fedavgpy](https://github.com/lx10077/fedavgpy) and [FedProx](https://github.com/litian96/FedProx).


## Setup network for experiment
1. Turn on Wifi router
2. Connect to Wifi
   
3. Config Wifi

    3.1. Open browser, and enter URL

    3.2. PORT Management -> DHCP Setting

        3.3.1. Scan devices under this wifi
        3.3.2. Check device's connect by identity

    3.3. VLAN: address binding
        
        3.4.1. Address of server to a fix ip 
        3.4.2. Tips: fix ip of devices and note them

## do experiment

### Overview
on server
`
python app.py --model server
`

on client
`
python app.py --model client
`

1. Preparation
    1.1. add ssh-key, remove existing files, send latest code 
    ```
    sh prepare.sh
    ```
    1.2. Generate data by following code and create `log/` folder. More help information could be found in [FedProx](<https://github.com/litian96/FedProx>).

    ```
    cd Stackelberg
    python data/synthetic/generate_synthetic.py
    python data/mnist/generate_random_niid.py
    python data/emnist/generate_random_niid.py
        ``` 
    

    1.3. run `Pre.py` to fetch alpha for different dataset 

2. To compare benchmark, activate server `python main_bench.py --model server`
   To compare properties, activate server  `python main_property.py --model server`
   
3. execute code on clients

    3.1.  Modify USERNAME, HOSTS and other parameters in `run.sh` and run `sh run.sh`

    
Notes:
1.In `args` of `Pre.py main_bench.py main_property.py`, you can modify following parameters to get different results
'''
'dataset': dataset name
'test_num': test number
'C': cost
'budget': budget
'v': intrinsic value
'experiment_folder': folder to save result
'num_round': number of simulation
'alpha': alpha
'''
Results in .json format contains global accuary, loss, time and other imformation. 

2.Different dataset requires different parameters setting so that the solver can work sucessfully, e.g., too large cost on average may result in no solution(negative q) due to the constraint.    

