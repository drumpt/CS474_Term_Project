import os
import json

import pandas as pd

def get_json_list_from_data_dir(data_dir):
    '''
    Parameters
    ----------
    data_dir: directory where json files exist

    Return
    ----------
    json_list: list contains all directories of json files in data_dir

    '''

    json_list = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            json_list.append(os.path.join(data_dir, file))
    return json_list

def get_dataframe_from_json_list(json_list):
    '''
    Parameters
    ----------
    json_list: list contains all directories of json files

    Return
    ----------
    df: pandas dataframe extracted and concatenated from json files in json_list

    '''
    
    df_list = [] # first use list for memory efficiency
    for json_dir in json_list:
        with open(json_dir, 'r') as f:
            data = json.load(f)
        df_list.append(pd.DataFrame.from_dict(data))
    df = pd.concat(df_list).reset_index(drop = True)
    return df