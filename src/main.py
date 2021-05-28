import json

import argparse

import utils
import preprocess
import lda_test
import clustering
import information_retrieval

def main(config):
    json_list = utils.get_json_list_from_data_dir(config["data_dir"])
    df = utils.get_dataframe_from_json_list(json_list)

    # lda = lda_test.LDA(config["data_dir"], config["lda"]["num_trends"])
    # issue_list = lda.get_topics()
    issue_list = ['North Korea', 'Pyeongchang Olympic'] # hard-coded issues

    vectorizer = clustering.Vectorizer(df, config)
    vectorizer.train()
    vectorized_df = vectorizer.vectorize()
    print(vectorized_df)

    clusterizer = clustering.Clustering(vectorized_df, config)
    clustered_df = clusterizer.apply_clustering()

    # ir = information_retrieval.OnIssueEventTracking(vectorized_df, issue_list, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, required = False, default = "config.json")
    args = parser.parse_args()

    with open(args.config_dir) as config_file:
        config = json.load(config_file)
    main(config)