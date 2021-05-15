import argparse
import json

import preprocess
import utils

def main(config):
    json_list = preprocess.get_json_list_from_data_dir(config["data_dir"])
    df = preprocess.get_dataframe_from_json_list(json_list)
    # print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, required = False, default = "config.json")
    args = parser.parse_args()

    with open(args.config_dir) as config_file:
        config = json.load(config_file)
    main(config)