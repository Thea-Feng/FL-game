# Incentive Mechanism Design for Federated Learning with Partial Client Participation
# Part 1 Hardware prototype: Fed-IoT-Sci
**This code is used for randomly selecting clients for Federated Learning IoT experiment**



## Setup network for experiment
1. Turn on Wifi router
2. Connect to Wifi
   
3. Config Wifi

    3.1. Open browser, and enter URL

    3.2. user: xxxx pwd: xxxxx

    3.3. PORT Management -> DHCP Setting

        3.3.1. Scan devices under this wifi
        3.3.2. Check device's connect by identity

    3.4. VLAN: address binding
        
        3.4.1. Bing address of server to a fix ip 
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

### Adaptive Sampling FL-IoT workflow

1. Preparation
    1.1. add ssh-key, remove existing files, send latest code 
    ```
    sh prepare.sh
    ```
    1.2. Generate data and create `log/` folder

2. activate server `python app.py --model server`
   
3. execute code on clients

    3.1.   `sh run.sh`




# Part 2 Software code: just use computer to simulate

This repository contains the codes for the paper
> [Incentive Mechanism Design for Federated Learning with Partial Client Participation]



Our code is based on the code for [fedavgpy](https://github.com/lx10077/fedavgpy) and [FedProx](https://github.com/litian96/FedProx).


## Usage

1. First generate data by the following code. Here `generate_random_niid` is used to generate the dataset named as ` mnist unbalanced ` in our paper,  where the number of samples among devices follows a power law. More help information could be found in [FedProx](<https://github.com/litian96/FedProx>).

   ```
   cd Stackelberg
   python data/synthetic/generate_synthetic.py
   python data/mnist/generate_random_niid.py
   python data/emnist/generate_random_nidd.py
    ```
Then run `Pre.py` to fetch alpha (running instruction is the same with step 2). You can change two probolities of uniformly sampling for different dataset. Paste the result and its dataset name to `alpha_utils.py`.      
   

2. Then start to train. You can run a single algorithm on a specific configuration like

    ```
   cd Stackelberg
   python $MAIN --device $DEVICE --dataset $DATASET --without_r False --num_round $T --real_round $N --num_epoch $E --batch_size $B --lr $LR  --seed $SEED --model $NET --algo $ALGO  --noaverage --noprint --test_num $P
   ```
    ```
    MAIN: To compare benchmark: main_$DATATYPE_bench.py 
    To compare property: main_$DATATYPE_test.py
    DATATYPE: synthetic, mnist, emnist
    DEVICE: use which gpu
    T: number of rounds to simulate (for Pre.py, set T = N = 50)
    N: number of rounds to calculate particpation level (Usually, it equals T. If you just want to get the result of stackelberg game in a short time and do not want to train data, then let T = 1, N be the number of your assumption
    P: number to distinguish result files

You can change model, budget, cost and other parameters in main function in $MAIN. To get more details, please refer to the code annotation.

3. The result files are stored under test-result folder. You can compare results or draw graphs you need.
    
Notes:
Different dataset requires different parameters setting so that the solver can work sucessfully, e.g., too large cost may result in no solution(negative q) due to the constraint.    
