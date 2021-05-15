import os
import string
import json

import nltk

def tokenization(self, sentence):
    return nltk.tokenize.word_tokenize(sentence)

def remove_punctuation(self, words):
    return [word for word in words if not word in string.punctuation and not word[0].isdigit()]

def identifier_normalization(self, words):
    normalized_words = []
    for word in words:
        if self.snake_case_breakdown(word)[0] != word:
            normalized_words.extend(self.snake_case_breakdown(word))
        elif word.isalnum() and self.camel_case_breakdown(word)[0] != word:
            normalized_words.extend(self.camel_case_breakdown(word))
        else:
            normalized_words.append(word)
    return normalized_words

def case_normalization(self, words):
    return [word.lower() for word in words]

def stop_word_filtering(self, words):
    stop_words = nltk.corpus.stopwords.words("english")
    return [word for word in words if not word in stop_words]

def stemming(self, words):
    snowball = nltk.stem.SnowballStemmer("english")
    return [snowball.stem(word) for word in words]

def snake_case_breakdown(self, identifier):
    return identifier.split("_")

def camel_case_breakdown(self, identifier):
    idx = list(map(str.isupper, identifier))
    l = [0]
    for (i, (x, y)) in enumerate(zip(idx, idx[1:])):
        if x and not y: # "Ul"
            l.append(i)
        elif not x and y: # "lU"
            l.append(i+1)
    l.append(len(identifier))
    return [identifier[x:y] for x, y in zip(l, l[1:]) if x < y]