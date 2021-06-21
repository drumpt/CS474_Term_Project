import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import pickle
import argparse
import random
import logging
logging.getLogger("gensim").setLevel(logging.ERROR)

import utils
import preprocess
import lda_test
import clustering
import clustering_section
import information_retrieval


def main(config):
    json_list = utils.get_json_list_from_data_dir(config["data_dir"])
    df = utils.get_dataframe_from_json_list(json_list)
    with open(config["vectorize"]["inverted_index_dir"], "rb") as f:
        inverted_index = pickle.load(f)
    # issue_selector = utils.IssueSelector(config, inverted_index)
    issue_list = ["Japanese military sexual slavery agreement between Korea and Japan", "North Korea Nuclear Test", "pyeongchang olympic"] # selected issues

    document_filter = preprocess.DocumentFilter(df, issue_list, inverted_index)
    filtered_df = document_filter.apply_filtering()

    vectorizer = clustering.Vectorizer(filtered_df, config)
    vectorized_df = vectorizer.vectorize()

    clusterizer = clustering.Clustering(vectorized_df, config)
    clustered_df = clusterizer.apply_clustering()
    # clusterizer.evaluate()

    tfidf_vectorizer = information_retrieval.InformationRetrieval(clustered_df, inverted_index, config)
    final_df = tfidf_vectorizer.get_tfidf_vector_list()

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