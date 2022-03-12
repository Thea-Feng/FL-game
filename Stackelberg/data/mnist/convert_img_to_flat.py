import torch
import numpy as np
import json
import os
from torch.functional import Tensor
import torchvision
cpath = os.path.dirname(__file__)
import torchvision.transforms as transforms



def convert_json_2_dict(file_name, file_path='.'):
    file = os.path.join(file_path, file_name)
    with open(file) as fp:
        dictContent = json.load(fp)
    return dictContent


def convert_dict_2_json(dictContent, file_name, file_path='.'):
    file = os.path.join(file_path, file_name)
    with open(file, 'w') as fp:
        json.dump(dictContent, fp)
        

file_name_img = 'niid1_6_1_N100_v1.0.json'
file_name_fla = 'niid1_6_0_N100_v1.0.json'

for dir in ['data/test','data/train']:
    path = os.path.join(dir, file_name_img)
    dict = convert_json_2_dict(path)

    for u in dict['user_data'].keys():
        # for i, _ in enumerate(dict['user_data'][u]['x']):
        print(np.array(dict['user_data'][u]['x']).shape)
        dict['user_data'][u]['x'] = (Tensor(dict['user_data'][u]['x']).view(-1, 784).numpy()/255).tolist()
        print(np.array(dict['user_data'][u]['x']).shape)
    pathou = os.path.join(dir, file_name_fla)
    convert_dict_2_json(dict,pathou)

