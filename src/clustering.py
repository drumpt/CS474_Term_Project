import random

import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import DBSCAN

import preprocess

class Vectorizer:
    def __init__(self, df, config):
        self.preprocessor = preprocess.Preprocessor()
        self.df = df
        self.df['preprocessed_body'] = df["body"].apply(self.preprocessor.preprocess)

        # TODO: consider title, time, section
        # TODO: save best model
        self.tagged_body = [TaggedDocument(words = row['preprocessed_body'], tags = [row['id']]) for index, row in self.df.iterrows()]
        self.model = Doc2Vec(
            window = 10,
            vector_size = 1024,
            alpha = 0.025,
            min_count = 2,
            dm = 1,
            negative = 5,
            seed = config["doc2vec"]["random_seed"]
        )
        self.model.build_vocab(self.tagged_body)
        self.epoch = config["doc2vec"]["epoch"]

    def train(self):
        for epoch in range(self.epoch):
            print(f"epoch {epoch}")
            self.model.train(self.tagged_body, total_examples = self.model.corpus_count, epochs = self.epoch)
            self.model.alpha *= 0.9

        self.df['vectorized_body'] = self.df['preprocessed_body'].apply(self.model.infer_vector)
        return self.df

    def predict(self, body):
        result = self.model.infer_vector(self.preprocessor.preprocess(body))
        return result

class Clustering:
    def __init__(self, df, config):
        self.df = df
        self.weight = config["clustering"]["weight"]
        self.method = config["clustering"]["method"]

    def apply_clustering(self):
        n_classes = {}

        if self.method == "DBSCAN":
            for i in np.arange(0.0001, 1, 0.0002):
                dbscan = DBSCAN(eps = i, min_samples = 1, metric = 'cosine').fit(self.df["vectorized_body"].tolist())
                n_classes.update({i: len(pd.Series(dbscan.labels_).value_counts())})
        
            dbscan = DBSCAN(eps = 0.15, min_samples = 1, metric = 'cosine').fit(self.df["vectorized_body"].tolist())
            self.df["cluster_number"] = dbscan.labels_

        # return self.df

        print(n_classes)