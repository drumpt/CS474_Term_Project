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


from doc2vec.data import doc, batch_dmsec, batch_dm
from doc2vec.model import model, dmsec, dm
from doc2vec import vocab

class Vectorizer:
    def __init__(self, df, config, sec=1):
        self.preprocessor = preprocess.Preprocessor()
        self.df = df
        self.df["preprocessed_body"] = df["body"].apply(self.preprocessor.preprocess)
        self.df["preprocessed_title"] = df["title"].apply(self.preprocessor.preprocess)
        self.df["preprocessed_section"] = df["section"].apply(self.section_to_id)

        self.tagged_body = [TaggedDocument(words = row["preprocessed_body"], tags = [row["id"]]) for _, row in self.df.iterrows()]
        self.part_weight = {
            "title": 0,
            "body": 1,
            "section": 0
        }

        # hyperparameters for doc2vec training
        self.epoch = 50
        self.output_dir = "weights"
        self.weight_dir = "models/dm_weights_049_0.0000.hdf5"
        self.callback = Callback(output_dir = self.output_dir)
        self.v = vocab.Vocabulary()
        all_tokens = []
        for body in enumerate(self.df["preprocessed_body"]):
            all_tokens += body[1]
        self.v.build(all_tokens, max_size=vocab.DEFAULT_SIZE)
        if sec:
            self.section = True
            self.model = dmsec.DMSEC(10, self.v.size, len(self.df["body"]),
                            embedding_size=256)

            self.generator = batch_dmsec.data_generator
            self.batch = batch_dmsec.batch
        else:
            self.section = False
            self.model = dm.DM(10, self.v.size, len(self.df["body"]),
                            embedding_size=256)

            self.generator = batch_dm.data_generator
            self.batch = batch_dm.batch




    def train(self):
        if not os.path.exists(self.weight_dir):
            self.model.build()
            self.model.compile()
        else:
            self.model.load(self.weight_dir)
        elapsed_epochs = 0
        token_ids_by_doc_id = {id: self.v.to_ids(token) for id, token in enumerate(self.df["preprocessed_body"])}
        section_ids_by_doc_id = {id: section for id, section in enumerate(self.df["preprocessed_section"])}
        if self.section:
            all_data = self.batch(
                            self.generator(
                                token_ids_by_doc_id,
                                10,
                                self.v.size, section_ids_by_doc_id))
        else:
            all_data = self.batch(
                            self.generator(
                                token_ids_by_doc_id,
                                10,
                                self.v.size))

        history = self.model.train(all_data,
                    epochs=self.epoch,
                    steps_per_epoch=model.DEFAULT_STEPS_PER_EPOCH,
                    early_stopping_patience=None,
                    save_path="models/dm_weights_099_0.0000.hdf5",
                    save_period=None,
                    save_doc_embeddings_path=None,
                    save_doc_embeddings_period=None)

        elapsed_epochs = len(history.history['loss'])
        self.model.save(
            "models/dm_weights_099_0.0000.hdf5".format(epoch=elapsed_epochs))

    def vectorize(self):
        if not os.path.exists("models/dm_weights_099_0.0000.hdf5"):
            raise Exception(f"File not found: {self.weight_dir}")
        else:
            self.model.load("models/dm_weights_099_0.0000.hdf5")

        def time_to_timestamp(t):
            return time.mktime(datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timetuple())

        def normalize(x):
            x_list = []
            for i in range(len(x)):
                try:
                    x_list.append(x[i][1])
                except: # vectorized_time
                    x_list.append(x.iloc[i])
            x = np.array(x_list, dtype = float)
            return (x - np.mean(x, axis = 0, keepdims = True)) / np.std(x, axis = 0, keepdims = True)

        # TODO: calculate section vector
        if (self.section):
            self.df["vectorized_section"] = [(i, self.model.section_embeddings[section]) for i, section in enumerate(self.df["preprocessed_section"])]
        self.df["vectorized_body"] = [(i, vector) for i, vector in enumerate(self.model.doc_embeddings)]
        self.df["vectorized_time"] = self.df["time"].apply(time_to_timestamp)

        if self.section:
            vectorized_full_text = self.part_weight["body"] * normalize(self.df["vectorized_body"]) \
                + self.part_weight["section"] * normalize(self.df["vectorized_section"])
        else:
            vectorized_full_text = self.part_weight["body"] * normalize(self.df["vectorized_body"])
        vectorized_time = np.expand_dims(normalize(self.df['vectorized_time']), axis = 1)


        self.df["vector"] = pd.Series(np.concatenate((vectorized_full_text, vectorized_time), axis = 1).tolist())
        return self.df

    def section_to_id(self, section):
        sec = ['North Korea', 'Social affairs', 'Defense', 'Foreign Policy', 'Diplomatic Circuit', 'Politics', 'Foreign  Affairs', 'National', 'Science', 'Education', 'International']
        id = 11
        for i in range(len(sec)):
            if sec[i] in section:
                id = i
        return id


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