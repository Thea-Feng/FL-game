import os 
import json

def convert_dict_2_json(dictContent, file_name, file_path='cache/'):
    file = os.path.join(file_path, file_name)
    with open(file, 'w') as fp:
        json.dump(dictContent, fp)

def convert_json_2_dict(file_name, file_path='cache/'):
    file = os.path.join(file_path, file_name)
    with open(file) as fp:
        dictContent = json.load(fp)

    return dictContent