import os
import json
import pickle

import argparse

import utils
import preprocess
import lda_test
import clustering
import clustering_section
import information_retrieval

def main(config):
    json_list = utils.get_json_list_from_data_dir(config["data_dir"])
    df = utils.get_dataframe_from_json_list(json_list)

    if config["vectorize"]["recalculation"]:
        vectorizer = clustering.Vectorizer(df, config)
        vectorized_df = vectorizer.vectorize()

        inverted_index_calculator = clustering.InvertedIndex(df, config)
        inverted_index = inverted_index_calculator.make_inverted_index()

        clusterizer = clustering.Clustering(vectorized_df, config)
        clustered_df = clusterizer.apply_clustering()

        tfidf_vectorizer = information_retrieval.InformationRetrieval(clustered_df, inverted_index, config)
        final_df = tfidf_vectorizer.get_tfidf_vector_list()
    else:
        if os.path.exists(config["vectorize"]["final_dataframe_dir"]):
            with open(config["vectorize"]["final_dataframe_dir"], "rb") as f:
                final_df = pickle.load(f)
        if os.path.exists(config["vectorize"]["inverted_index_dir"]):
            with open(config["vectorize"]["inverted_index_dir"], "rb") as f:
                inverted_index = pickle.load(f)

    # lda = lda_test.LDA(config["data_dir"], config["lda"]["num_trends"])
    # issue_list = lda.get_topics()
    issue_list = ['Pyeongchang Olympic', 'North Korea nuclear test'] # hard-coded issues

    def filter_article_by_issue(row):
        for issue in issue_list:
            if issue.lower() in row['body'].lower():
                return True
        return False

    on_issue_df = df[df.apply(filter_article_by_issue, axis = 1)].copy()

    print(df.dtypes)
    print(on_issue_df.dtypes)

    # vectorizer = clustering.Vectorizer(on_issue_df, config)
    vectorizer = clustering.Vectorizer(df, config)
    #vectorizer = clustering_section.Vectorizer(df, config, sec=0)
    #vectorizer.train()
    vectorized_df = vectorizer.vectorize()

    print("vectorizing done")

    #print(vectorized_df.head())

    #print(vectorized_df)

    clusterizer = clustering.Clustering(vectorized_df, config)
    #clusterizer = clustering_section.Clustering(vectorized_df, config)
    clustered_df = clusterizer.apply_clustering()

    print("clustering done")

    #on_issue_event_tracker = information_retrieval.OnIssueEventTracking(clustered_df, issue_list, config)
    #on_issue_event_tracker.apply_on_issue_event_tracking()

    print("on issue event tracking done")

    related_issue_event_tracker = information_retrieval.RelatedIssueEventTracking(final_df, inverted_index, issue_list, config)
    related_issue_event_tracker.apply_related_issue_event_tracking()
    print("related issue event tracking done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, required = False, default = "config.json")
    args = parser.parse_args()

    with open(args.config_dir) as config_file:
        config = json.load(config_file)
    main(config)