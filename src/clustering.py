import os
import random
import time
from datetime import datetime

import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import DBSCAN

import preprocess

class Vectorizer:
    def __init__(self, df, config):
        self.preprocessor = preprocess.Preprocessor()
        self.df = df
        self.df['preprocessed_body'] = df['body'].apply(self.preprocessor.preprocess)
        self.df['preprocessed_title'] = df['title'].apply(self.preprocessor.preprocess)

        self.vector_weight = config["doc2vec"]["weight"]
        self.output_dir = config["doc2vec"]["output_dir"]

        self.callback = Callback(output_dir = self.output_dir)

        self.epoch = config["doc2vec"]["epoch"]
        self.tagged_body = [TaggedDocument(words = row['preprocessed_body'], tags = [row['id']]) for _, row in self.df.iterrows()]
        self.model = Doc2Vec(
            window = 10,
            vector_size = 256,
            alpha = 0.025,
            min_count = 2,
            dm = 1,
            negative = 5,
            seed = config["doc2vec"]["random_seed"],
            compute_loss = True
        )
        self.model.build_vocab(self.tagged_body)


    def train(self):
        self.model.train(
            self.tagged_body,
            total_examples = self.model.corpus_count,
            epochs = self.epoch,
            callbacks = [self.callback]
        )

    def vectorize(self):
        self.df['vectorized_body'] = self.df['preprocessed_body'].apply(self.model.infer_vector)
        self.df['vectorized_title'] = self.df['preprocessed_title'].apply(self.model.infer_vector)

        # TODO: consider section vector
        self.df['vectorized_section'] = 0

        time_to_timestamp = lambda t: time.mktime(datetime.strptime(t, '%Y-%m-%d %H:%M:%S').timetuple())
        scaler = lambda x: MinMaxScaler().fit_transform(x.reshape(-1, 1))
        self.df['vectorized_time'] = self.df['time'].apply(time_to_timestamp).apply(scaler)

        # self.df['weighted_vector'] = self.weight['title'] * self.df['vectorized_title'] \
        #     + self.vector_weight['body'] * self.df['vectorized_body'] \
        #     + self.vector_weight['section'] * self.df['vectorized_section'] \
        #     + self.vector_weight['time'] * self.df['vectorized_time']
        # return self.df

    # def predict(self, body):
    #     result = self.model.infer_vector(self.preprocessor.preprocess(body))
    #     return result


class Callback(CallbackAny2Vec):
    def __init__(self, output_dir):
        self.epoch = 0
        self.output_dir = output_dir

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        model.save(os.path.join(self.output_dir, f"weights_{self.epoch:03d}_{loss:.4f}.gz"))
        print(f'Epoch : {self.epoch} loss : {loss}')
        self.epoch += 1


class Clustering:
    def __init__(self, df, config):
        self.df = df
        self.weight = config["clustering"]["weight"]
        self.method = config["clustering"]["method"]

    def apply_clustering(self):
        pass
        # n_classes = {}

        # if self.method == "DBSCAN":
        #     for i in np.arange(0.0001, 1, 0.0002):
        #         dbscan = DBSCAN(eps = i, min_samples = 1, metric = 'cosine').fit(self.df["vectorized_body"].tolist())
        #         n_classes.update({i: len(pd.Series(dbscan.labels_).value_counts())})
        
        #     dbscan = DBSCAN(eps = 0.15, min_samples = 1, metric = 'cosine').fit(self.df["vectorized_body"].tolist())
        #     self.df["cluster_number"] = dbscan.labels_

        # # return self.df

        # print(n_classes)