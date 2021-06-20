import os
import json
import pickle
import argparse

import utils
import preprocess
import lda_test
import clustering
import information_retrieval

def main(config):
    json_list = utils.get_json_list_from_data_dir(config["data_dir"])
    df = utils.get_dataframe_from_json_list(json_list)
    print("load data done!")

    if config["vectorize"]["recalculation"]:

        vectorizer = clustering.Vectorizer(df, config)
        vectorized_df = vectorizer.vectorize()
        print("vectorize done!")

        inverted_index_calculator = clustering.InvertedIndex(df, config)
        inverted_index = inverted_index_calculator.make_inverted_index()

        clusterizer = clustering.Clustering(vectorized_df, config)
        clustered_df = clusterizer.apply_clustering()
        print("clustering done!")

        tfidf_vectorizer = information_retrieval.InformationRetrieval(clustered_df, inverted_index, config)
        final_df = tfidf_vectorizer.get_tfidf_vector_list()
        print("final on-issue tracking done!")

    else:
        if os.path.exists(config["vectorize"]["final_dataframe_dir"]):
            with open(config["vectorize"]["final_dataframe_dir"], "rb") as f:
                final_df = pickle.load(f)
        if os.path.exists(config["vectorize"]["inverted_index_dir"]):
            with open(config["vectorize"]["inverted_index_dir"], "rb") as f:
                inverted_index = pickle.load(f)

    issue_list = ['North Korea Nuclear Test', 'Pyeongchang Olympic'] # hard-coded issues

    on_issue_event_tracker = information_retrieval.OnIssueEventTracking(final_df, inverted_index, issue_list, config)
    on_issue_event_tracker.apply_on_issue_event_tracking()

    related_issue_event_tracker = information_retrieval.RelatedIssueEventTracking(final_df, inverted_index, issue_list, config)
    related_issue_event_tracker.apply_related_issue_event_tracking()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, required = False, default = "config.json")
    args = parser.parse_args()

    with open(args.config_dir) as config_file:
        config = json.load(config_file)
    main(config)