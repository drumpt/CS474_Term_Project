import os
import random
import time
from datetime import datetime

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering

import preprocess

class Vectorizer:
    def __init__(self, df, config):
        self.preprocessor = preprocess.Preprocessor()
        self.df = df

        print("vectorizing 1")

        self.df["preprocessed_body"] = df["body"].apply(self.preprocessor.preprocess)

        print("vectorizing 2")

        self.df["preprocessed_title"] = df["title"].apply(self.preprocessor.preprocess)

        print("vectorizing 3")

        self.tagged_body = [TaggedDocument(words = row["preprocessed_body"], tags = [row["id"]]) for _, row in self.df.iterrows()]
        self.part_weight = config["doc2vec"]["part_weight"]

        # hyperparameters for doc2vec training
        self.epoch = config["doc2vec"]["epoch"]
        self.output_dir = config["doc2vec"]["output_dir"]
        self.weight_dir = config["doc2vec"]["weight_dir"]
        self.callback = Callback(output_dir = self.output_dir)
        self.model = Doc2Vec(
            window = 10,
            vector_size = config["doc2vec"]["embedding_dim"],
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
            callbacks = [self.callback],
            compute_loss = True
        )

    def vectorize(self):
        if not os.path.exists(self.weight_dir):
            raise Exception(f"File not found: {self.weight_dir}")
        else:
            self.model = Doc2Vec.load(self.weight_dir)

        def time_to_timestamp(t):
            return time.mktime(datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timetuple())

        def normalize(x):
            x_list = []
            for i in range(len(x)):
                try:
                    x_list.append(list(x[i]))
                except: # vectorized_time
                    x_list.append(x[i])
            x = np.array(x_list, dtype = float)
            return (x - np.mean(x, axis = 0, keepdims = True)) / np.std(x, axis = 0, keepdims = True)

        self.df["vectorized_title"] = self.df["preprocessed_title"].apply(self.model.infer_vector)
        # TODO: calculate section vector
        self.df["vectorized_section"] = self.df["preprocessed_body"].apply(self.model.infer_vector)
        self.df["vectorized_body"] = self.df["preprocessed_body"].apply(self.model.infer_vector)
        self.df["vectorized_time"] = self.df["time"].apply(time_to_timestamp)

        vectorized_full_text = self.part_weight["title"] * normalize(self.df["vectorized_title"]) \
            + self.part_weight["body"] * normalize(self.df["vectorized_body"]) \
            + self.part_weight["section"] * normalize(self.df["vectorized_section"])
        vectorized_time = np.expand_dims(normalize(self.df['vectorized_time']), axis = 1)

        self.df["vector"] = pd.Series(np.concatenate((vectorized_full_text, vectorized_time), axis = 1).tolist())
        return self.df


class Callback(CallbackAny2Vec):
    def __init__(self, output_dir):
        self.epoch = 0
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        model.save(os.path.join(self.output_dir, f"weights_{self.epoch:03d}_{loss:.4f}.h5"))
        print(f"Epoch : {self.epoch} loss : {loss}")
        self.epoch += 1


class Clustering:
    def __init__(self, df, config):
        self.df = df
        self.method = config["clustering"]["method"]

    def apply_clustering(self):

        if self.method == "DBSCAN":
            # experiemnt
            for i in np.arange(0.001, 1, 0.001):
                clusterizer = DBSCAN(eps = i, min_samples = 1, metric = "cosine").fit(self.df["vector"].tolist())
                print(i, len(pd.Series(clusterizer.labels_).value_counts()))

            clusterizer = DBSCAN(eps = 0.1, min_samples = 1, metric = "cosine").fit(self.df["vector"].tolist())

        elif self.method == "hierarchical":
            # # experiemnt
            # for i in np.arange(0.01, 100, 0.01):
            #     clusterizer = AgglomerativeClustering(n_clusters = None, distance_threshold = i).fit(self.df["vector"].tolist())
            #     print(i, len(pd.Series(clusterizer.labels_).value_counts()))

            clusterizer = AgglomerativeClustering(n_clusters = None, distance_threshold = 5).fit(self.df["vector"].tolist())

        elif self.method == "OPTICS":
            # # experiemnt
            # for i in np.arange(1, 100):
            #     clusterizer = OPTICS(eps = i, min_samples = 2).fit(self.df["vector"].tolist())
            #     print(i, len(pd.Series(clusterizer.labels_).value_counts()))

            clusterizer = OPTICS(eps = 0.1, min_samples = 2).fit(self.df["vector"].tolist())
        else:
            raise Exception(f"Invalid: {self.weight_dir}")            
        
        self.df["cluster_number"] = clusterizer.labels_
        return self.df

    # TODO: need to evaluate various clustering algorithms
    def evaluate(self):
        pass