import logging
import os
import re
import json
import pandas

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def _read(path):
    with open(path, 'r') as f:
        return f.read()


def _doc_id(path):
    doc_id, = re.search(r'(\d+)', path).groups()
    return doc_id


def _full_paths(directory):
    return [os.path.join(directory, path) for path in os.listdir(directory)]


def docs_by_id(directory):
    # logger.info('Loading documents from {}'.format(directory))
    return {_doc_id(path): _read(path) for path in _full_paths(directory)}


def tokens(doc):
#    return word_tokenize(doc)

    stop_words = set(stopwords.words()) 
    tokenized = word_tokenize(doc)
   
    alpha_tokens = [t.lower() for t in tokenized if t.isalpha()]
    no_stops_tokens = [t for t in alpha_tokens if t not in stop_words]

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(t) for t in no_stops_tokens]
    return lemmatized_tokens


def tokens_by_doc_id(path, num):
#    nltk.download('punkt')
#    return {doc_id: tokens(doc) for doc_id, doc in docs_by_id(directory).items()}
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    datas = []
    for i in range(num):
        with open( path + "/koreaherald_1517_" + str(i) + ".json", 'r') as f:
            datas += [pandas.DataFrame.from_dict(json.load(f))]

    df = pandas.concat(datas)
    ret = {}
    for i in range (len(df.index)):
        ret[i] = tokens(df[' body'][i])
    return ret

