import argparse
import itertools

from numpy.lib.shape_base import tile

from doc2vec.data import batch_dm, batch_dbow, doc, batch_dmsec
from doc2vec.model import dm, dbow, model, dmsec
from doc2vec import vocab

import json
import pandas
import math

import tensorflow as tf


MODEL_TYPES = {
    'dm': (dm.DM, batch_dm.data_generator, batch_dm.batch),
    'dbow': (dbow.DBOW, batch_dbow.data_generator, batch_dbow.batch),
    'dmsec': (dmsec.DMSEC, batch_dmsec.data_generator, batch_dmsec.batch)
}
def sections_by_id(num):
    #return {doc_id: tokens(doc) for doc_id, doc in docs_by_id(directory).items()}
    datas = []
    for i in range(num):
        with open( "../../dataset/koreaherald_1517_" + str(i) + ".json", 'r') as f:
            datas += [pandas.DataFrame.from_dict(json.load(f))]

    df = pandas.concat(datas)
    sections = {}
    titles = {}
    for i in range (len(df.index)):
        sections[i] = df[' section'][i]
        titles[i] = df['title'][i]
    return sections, titles


def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx * sumyy)

def euclidean_distance(v1, v2):
    return tf.norm(v1-v2, ord='euclidean')
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return math.sqrt(sumxx - 2 * sumxy + sumyy)

def jaccard_distance(v1, v2):
    tp = tf.reduce_sum(tf.mul(v1, v2), 1)
    fn = tf.reduce_sum(tf.mul(v1, 1-v2), 1)
    fp = tf.reduce_sum(tf.mul(1-v1, v2), 1)
    return 1 - (tp / (tp + fn + fp))

def dot_product(v1, v2):
    return tf.tensordot(v1, v2, 1)


def main():

    tokens_by_doc_id = doc.tokens_by_doc_id("../../dataset", 1)

    num_docs = len(tokens_by_doc_id)

    v = vocab.Vocabulary()
    v.load("./total_vocab.vocab")
    #token_ids_by_doc_id = {d: v.to_ids(t) for d, t in tokens_by_doc_id.items()}
    
    #print(token_ids_by_doc_id[2])
    model_class, data_generator, batcher = MODEL_TYPES['dmsec']

    m = model_class(model.DEFAULT_WINDOW_SIZE, v.size, num_docs,
                    embedding_size=model.DEFAULT_EMBEDDING_SIZE)

    m.load("./total_150_dm.hdf5") 

    elapsed_epochs = 0
    #print(m.doc_embeddings.shape)
    #print(len(v._vocab))
    print(m.doc_embeddings.shape)

    sec_dic, title_dic = sections_by_id(1)

    secs = [[], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(3000):
        if sec_dic[i] == 'North Korea':
            secs[0] += [m.doc_embeddings[i]]
            #print("NK ", title_dic[i])
        elif sec_dic[i] == 'Social affairs':
            secs[1] += [m.doc_embeddings[i]]
            #print("sa " , title_dic[i])
        elif sec_dic[i] == 'Defense':
            secs[2] += [m.doc_embeddings[i]]
            #print("df " , title_dic[i])
        elif sec_dic[i] == 'Foreign Policy':
            secs[3] += [m.doc_embeddings[i]]
            #print("fp " , title_dic[i])
        elif 'Diplomatic Circuit' in  sec_dic[i]:
            secs[4] += [m.doc_embeddings[i]]
            #print("dc " , title_dic[i])
        elif sec_dic[i] == 'Politics':
            secs[5] += [m.doc_embeddings[i]]
            #print("po " , title_dic[i])
        elif sec_dic[i] == 'Foreign  Affairs':
            secs[6] += [m.doc_embeddings[i]]
            #print("po " , title_dic[i])
        elif sec_dic[i] == 'National':
            secs[7] += [m.doc_embeddings[i]]
            #print("po " , title_dic[i])
        elif sec_dic[i] == 'Science':
            secs[8] += [m.doc_embeddings[i]]
            #print("po " , title_dic[i])
        elif sec_dic[i] == 'Education':
            secs[9] += [m.doc_embeddings[i]]
            #print("po " , title_dic[i])
        elif sec_dic[i] == 'International':
            secs[10] += [m.doc_embeddings[i]]
            #print("po " , title_dic[i])
        else:
            secs[11] += [m.doc_embeddings[i]]
            #print("el ",  title_dic[i])

    #print((secs[1][0]).shape)
    #print(sec_dic[0], sec_dic[1])
    #print(cosine_similarity(m.doc_embeddings[0], m.doc_embeddings[1]))
    for i in range(12):
        sim = 0.0
        docnum = len(secs[i])
        print(i, docnum)
        for j in range(docnum):
            for k in range(j, docnum):
                sim += dot_product(secs[i][j], secs[i][k])
        print(i, sim / (docnum * (docnum - 1) / 2))

    total_sim = 0.0
    total_doc = len(m.doc_embeddings)
    for i in range(total_doc):
        for j in range(i, total_doc):
            total_sim += dot_product(m.doc_embeddings[i], m.doc_embeddings[j])
    print(i, total_sim / (total_doc * (total_doc - 1) / 2))
    """print(cosine_similarity(secs[0][1], secs[0][0]))
    print(cosine_similarity(secs[0][2], secs[0][0]))
    print(cosine_similarity(secs[0][3], secs[0][0]))
    print(cosine_similarity(secs[1][1], secs[0][0]))
    print(cosine_similarity(secs[1][1], secs[1][0]))

    if False:   #train
        all_data = batcher(
                data_generator(
                    token_ids_by_doc_id,
                    model.DEFAULT_WINDOW_SIZE,
                    v.size))

        history = m.train(
                all_data,
                epochs=model.DEFAULT_NUM_EPOCHS,
                steps_per_epoch=model.DEFAULT_STEPS_PER_EPOCH,
                early_stopping_patience=args.early_stopping_patience,
                #save_path=args.save,
                #save_period=args.save_period,
                save_doc_embeddings_path=args.save_doc_embeddings,
                save_doc_embeddings_period=args.save_doc_embeddings_period)

        elapsed_epochs = len(history.history['loss'])

    if False:   #save
        m.save(
            args.save.format(epoch=elapsed_epochs))

    if True:
        m.save_doc_embeddings(
            "./dm_embeddings.hdf5".format(epoch=elapsed_epochs))"""

if __name__=="__main__":
    main()