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
    df.columns = ['title', 'author', 'time', 'description', 'body', 'section']
    return df


def get_dataframe_from_json_list_by_year(json_list):
    '''
    Parameters
    ----------
    json_list: list contains all directories of json files

    Return
    ----------
    df_1: pandas dataframe of news articles at 2015
    df_2: pandas dataframe of news articles at 2016
    df_3: pandas dataframe of news articles at 2017

    '''

    df_list_1 = []  # first use list for memory efficiency
    df_list_2 = []
    df_list_3 = []
    for json_dir in json_list:
        with open(json_dir, 'r') as f:
            data = json.load(f)
        for data_index in range(len(data[" time"])):
            di = str(data_index)
            if data[" time"][di][:4] == "2015":
                new_data = {"title": data["title"][di], "author": data[" author"][di], "time": data[" time"][di],
                            "description": data[" description"][di], "body": data[" body"][di],
                            "section": data[" section"][di]}
                df_list_1.append(pd.DataFrame(new_data, index=[0]))
            elif data[" time"][di][:4] == "2016":
                new_data = {"title": data["title"][di], "author": data[" author"][di], "time": data[" time"][di],
                            "description": data[" description"][di], "body": data[" body"][di],
                            "section": data[" section"][di]}
                df_list_2.append(pd.DataFrame(new_data, index=[0]))
            elif data[" time"][di][:4] == "2017":
                new_data = {"title": data["title"][di], "author": data[" author"][di], "time": data[" time"][di],
                            "description": data[" description"][di], "body": data[" body"][di],
                            "section": data[" section"][di]}
                df_list_3.append(pd.DataFrame(new_data, index=[0]))

    # print(f"LEN IS {len(df_list_1)}")
    # print(f"LEN IS {len(df_list_2)}")
    # print(f"LEN IS {len(df_list_3)}")

    df_1 = pd.concat(df_list_1).reset_index(drop=True)
    df_1.columns = ['title', 'author', 'time', 'description', 'body', 'section']
    df_2 = pd.concat(df_list_2).reset_index(drop=True)
    df_2.columns = ['title', 'author', 'time', 'description', 'body', 'section']
    df_3 = pd.concat(df_list_3).reset_index(drop=True)
    df_3.columns = ['title', 'author', 'time', 'description', 'body', 'section']
    return df_1, df_2, df_3


if __name__ == "__main__":
    df_2015, df_2016, df_2017 = get_dataframe_from_json_list_by_year(get_json_list_from_data_dir("../dataset"))
