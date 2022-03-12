import torch
import numpy as np
import json
import os
import torchvision
cpath = os.path.dirname(__file__)
import torchvision.transforms as transforms

######### N 40 version

NUM_USER = 100
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = True  
np.random.seed(1)

NIID = (1,6)
VERSION = 'v4.0'

'''
    - version
        v1.0 seed 1
        v2.0 thd 500
        v3.0

'''
# Go to mnist/ then run the program


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)

def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst

def choose_two_digit(split_data_lst):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        print(available_digit)
    return lst


def main():
    # Get MNIST data, normalize, and divide by level
    print('>>> Get MNIST data.')
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                                ])

    trainset = torchvision.datasets.MNIST(DATASET_FILE, download=True, train=True, transform=transform)
    testset = torchvision.datasets.MNIST(DATASET_FILE, download=True, train=False, transform=transform)

    train_mnist = ImageDataset(trainset.train_data, trainset.train_labels)
    test_mnist = ImageDataset(testset.test_data, testset.test_labels)

    mnist_traindata = []
    for number in range(10):
        idx = train_mnist.target == number
        mnist_traindata.append(train_mnist.data[idx])

    mnist_testdata = []

    for number in range(10):
        idx = test_mnist.target == number
        mnist_testdata.append(test_mnist.data[idx])

    mnist_test_x = []
    mnist_test_y = []

    for number in range(10):
        mnist_test_x += mnist_testdata[number].tolist()
        mnist_test_y += (number*np.ones(len(mnist_testdata[number]))).tolist()
    
    num_samples = np.random.lognormal(4, 1.5, (NUM_USER)) + 50
    # num_samples = np.random.lognormal(4, 2, (NUM_USER+120)) + 50
    # num_samples = num_samples[10:-10]

    print("num_samples",sorted(num_samples.tolist()))

    idx = np.zeros(10, dtype=np.int64)
    cnt_num = np.zeros(10, dtype=np.int64)

    # Assign train samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    test_index = len(mnist_test_x) // NUM_USER 

    cnt_data_label = []

    for user in range(NUM_USER):

        if num_samples[user] < 100:
            num_class = np.random.randint(1,3)
        elif num_samples[user] > 1000:
            num_class = np.random.randint(5,7)
        else:
            num_class = np.random.randint(NIID[0], NIID[1] + 1)

        # print(user, num_samples[user], num_class)
        num_sample_per_class = int(num_samples[user] / num_class)

        cnt_data_label.append( (num_sample_per_class * num_class, num_class) )

        class_list = np.random.choice(10, num_class, replace=False).tolist()

        for class_id in class_list:
            if idx[class_id] + num_sample_per_class >= len(mnist_traindata[class_id]):
                idx[class_id] = 0

            train_X[user] += mnist_traindata[class_id][idx[class_id]: (idx[class_id] + num_sample_per_class)].tolist()
            train_y[user] += (class_id * np.ones(num_sample_per_class)).tolist()
            idx[class_id] += num_sample_per_class
            cnt_num[class_id] = max(idx[class_id], cnt_num[class_id])

        test_X[user] = mnist_test_x[:test_index]
        test_y[user] = mnist_test_y[:test_index]

        mnist_test_x = mnist_test_x[test_index:]
        mnist_test_y = mnist_test_y[test_index:]

    cnt_data_label.sort()
    print(np.array(cnt_data_label))
    print("cnt number", cnt_num)

    # Setup directory for train/test data
    print('>>> Set data path for MNIST.')
    image = 1 if IMAGE_DATA else 0
    str_niid = 'niid{}_{}'.format(NIID[0],NIID[1])
    if NIID[0] == NIID[1]:
        str_niid = 'niid{}'.format(NIID[0])
    train_path = './data/train/{}_{}_N{}_{}.json'.format(str_niid, image, NUM_USER, VERSION)
    test_path = './data/test/{}_{}_N{}_{}.json'.format(str_niid, image, NUM_USER, VERSION)

    print(train_path, test_path)
    # dir_path = os.path.dirname(train_path)
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)

    # dir_path = os.path.dirname(test_path)
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)

    # create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # setup users
    for i in range(NUM_USER):
        uname = i
        
        combined = list(zip(train_X[i], train_y[i]))
        np.random.shuffle(combined)
        train_X[i][:], train_y[i][:] = zip(*combined)
        num_samples = len(train_X[i])
        train_len = int(0.8 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': train_X[i][:train_len], 'y': train_y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': train_X[i][train_len:], 'y': train_y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    print('>>> User data distribution: {}'.format(sorted(train_data['num_samples'])))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'w') as outfile:
            json.dump(train_data, outfile)
        with open(test_path, 'w') as outfile:
            json.dump(test_data, outfile)

        print('>>> Save data.')


if __name__ == '__main__':
    main()
