import os
import string
import json
import re

import nltk

class Preprocessor:
    def __init__(self):
        pass

    def preprocess(self, body):
        '''
        tokenization -> remove_punctuation -> case_normalization -> stop_word_filtering -> stemming
        '''
        words = self.tokenization(body)
        words = self.remove_punctuation(words)
        words = self.case_normalization(words)
        words = self.stop_word_filtering(words)
        words = self.stemming(words)
        return words

    def tokenization(self, body):
        '''
        Break body into list of words
        '''
        return nltk.tokenize.word_tokenize(body)

    def remove_punctuation(self, words):
        '''
        Remove all the punctuations like !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ in words
        '''
        def match(word, punctuation_list):
            for punctuation in punctuation_list:
                if re.match(f"\{punctuation}+", word):
                    return True
            return False

        punctuation_list = string.punctuation
        return [word for word in words if not match(word, punctuation_list)]

    def case_normalization(self, words):
        '''
        'File' and 'FILE' will become 'file'
        '''
        return [word.lower() for word in words]

    def stop_word_filtering(self, words):
        '''
        Some words like 'a', 'an' are extremely common in English. Filter these words.
        '''
        stop_words = nltk.corpus.stopwords.words("english")
        return [word for word in words if not word in stop_words]

    def stemming(self, words):
        '''
        'opens' and 'opening' both become 'open'
        '''
        snowball = nltk.stem.SnowballStemmer("english")
        return [snowball.stem(word) for word in words]