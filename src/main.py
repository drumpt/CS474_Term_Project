import json

import argparse
import pandas as pd

import utils
import preprocess
import information_retrieval

def main(config):
    json_list = utils.get_json_list_from_data_dir(config["data_dir"])
    df = utils.get_dataframe_from_json_list(json_list)

    issue_list = ['north korea', 'pyeongchang olympic'] # hard-coded issues
    ir = information_retrieval.InformationRetrieval(df, issue_list, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, required = False, default = "config.json")
    args = parser.parse_args()

    with open(args.config_dir) as config_file:
        config = json.load(config_file)
    main(config)