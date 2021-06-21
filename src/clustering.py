import os
import random
import time
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
from sklearn import metrics

import preprocess

class Vectorizer:
    def __init__(self, df, config):
        self.preprocessor = preprocess.Preprocessor()
        self.df = df
        self.df["preprocessed_body"] = df["body"].apply(self.preprocessor.preprocess)
        self.df["preprocessed_title"] = df["title"].apply(self.preprocessor.preprocess)

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
        # self.model.init_sims(replace = True)
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
                    """print(x)
                    print(x.iloc[i])
                    print(type(x))"""
                    x_list.append(list(x.iloc[i]))
                except: # vectorized_time
                    """print("except")
                    print(x)
                    print(x.iloc[i])
                    print(type(x))
                    print(type(x.iloc[i]))"""
                    x_list.append(x.iloc[i])
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

        self.df["vector"] = np.concatenate((vectorized_full_text, vectorized_time), axis = 1).tolist()
        # print(self.df["vector"])
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


class InvertedIndex:
    def __init__(self, df, config):
        self.df = df
        self.output_dir = config["vectorize"]["inverted_index_dir"]
        if not os.path.exists(os.path.dirname(self.output_dir)):
            os.makedirs(os.path.dirname(self.output_dir))

    def make_inverted_index(self):
        inverted_index = dict()
        for _, row in self.df.iterrows():
            for word in row["preprocessed_body"]:
                if inverted_index.get(word):
                    if not row["id"] in inverted_index[word]:
                        inverted_index[word].append(row["id"])
                else:
                    inverted_index[word] = [row["id"]]

        with open(self.output_dir, "wb") as f:
            pickle.dump(inverted_index, f)
        # print("Finish making inverted index!")
        return inverted_index


class Clustering:
    def __init__(self, df, config):
        self.df = df
        self.method = config["clustering"]["method"]

    def apply_clustering(self):
        if self.method == "hierarchical":
            clusterizer = AgglomerativeClustering(n_clusters = None, distance_threshold = 10).fit(self.df["vector"].tolist())
        elif self.method == "DBSCAN":
            clusterizer = DBSCAN(eps = 10, min_samples = 1).fit(self.df["vector"].tolist())
        elif self.method == "OPTICS":
            clusterizer = OPTICS(eps = 10, min_samples = 2).fit(self.df["vector"].tolist())
        else:
            raise Exception(f"Invalid: {self.method}")

        self.df["cluster_number"] = clusterizer.labels_
        # print(self.df["cluster_number"])
        return self.df

    def evaluate(self):
        # hard-coded class-label
        class_number = [0, 1, 2, 3, 3, 4, 5, 6, 7, 5, 8, 9, 10, 11, 7, 12, 13, 14, 15, 7, 16, 13, 7, 17, 18, 13, 16, 19, 20, 17]
        self.evaluated_df = self.df.head(30)

        print(f"Evaluation result of {self.method} clustering")
        print("==================================================")
        print(f"Rand index : {metrics.rand_score(class_number, self.evaluated_df['cluster_number'].tolist())}")
        print(f"Adjusted rand index : {metrics.adjusted_rand_score(class_number, self.evaluated_df['cluster_number'].tolist())}")
        print(f"Mutual information : {metrics.adjusted_mutual_info_score(class_number, self.evaluated_df['cluster_number'].tolist())}")
        print(f"Homogeneity score : {metrics.homogeneity_score(class_number, self.evaluated_df['cluster_number'].tolist())}")
        print("\n\n\n")