import os
import string
import json

import nltk

def tokenization(body):
    '''
    Break body into list of words
    '''
    return nltk.tokenize.word_tokenize(body)

def remove_punctuation(words):
    '''
    Remove all the punctuations like "!()-[]{};:'"\,<>./?@#$%^&*_~" in words
    '''
    return [word for word in words if not word in string.punctuation]

def case_normalization(words):
    '''
    'File' and 'FILE' will become 'file'
    '''
    return [word.lower() for word in words]

def stop_word_filtering(words):
    '''
    Some words like 'a', 'an' are extremely common in English. Filter these words.
    '''
    stop_words = nltk.corpus.stopwords.words("english")
    return [word for word in words if not word in stop_words]

def stemming(words):
    '''
    'opens' and 'opening' both become 'open'
    '''
    snowball = nltk.stem.SnowballStemmer("english")
    return [snowball.stem(word) for word in words]

def preprocess_body(body):
    '''
    tokenization -> remove_punctuation -> case_normalization -> stop_word_filtering -> stemming
    '''
    words = tokenization(body)
    words = remove_punctuation(words)
    words = case_normalization(words)
    words = stop_word_filtering(words)
    words = stemming(words)
    return words