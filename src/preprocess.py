import os
import string
import json
import re

import nltk
nltk.download('punkt', quiet = True)
nltk.download('stopwords', quiet = True)


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
        stop_words.append('Yonhap')
        return [word for word in words if not word in stop_words]

    def stemming(self, words):
        '''
        'opens' and 'opening' both become 'open'
        '''
        snowball = nltk.stem.SnowballStemmer("english")
        return [snowball.stem(word) for word in words]

class DocumentFilter:
    def __init__(self, df, issue_list, inverted_index):
        self.df = df
        self.issue_list = issue_list
        self.inverted_index = inverted_index

    def get_candidate_id(self, issue_list):
        preprocessor = Preprocessor()
        candidate_id = set()
        for issue in issue_list:
            temp_candidate_id = set()
            is_first_token = True
            for word in preprocessor.preprocess(issue):
                if self.inverted_index.get(word):
                    if is_first_token:
                        temp_candidate_id = temp_candidate_id.union(self.inverted_index.get(word))
                        is_first_token = False
                    else:
                        temp_candidate_id = temp_candidate_id.intersection(self.inverted_index.get(word))
            candidate_id = temp_candidate_id.union(temp_candidate_id)
        print(len(candidate_id))
        return candidate_id
    
    def apply_filtering(self):
        candidate_id = self.get_candidate_id(self.issue_list)
        document_filter = lambda row: row['id'] in candidate_id
        return self.df[self.df.apply(document_filter, axis = 1)].copy()