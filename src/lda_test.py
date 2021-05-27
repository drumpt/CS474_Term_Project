import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.util import ngrams
import utils
import time


class LDA:
    def __init__(self, data_dir, num_trends):
        self.num_trends = num_trends
        self.dfs, self.tokenized_doc = self.preprocess(data_dir)
        print("ROUND 1 - Preprocess ended")
        self.dict, self.corpus = self.encoding(self.tokenized_doc)
        print("ROUND 2 - Encoding ended")
        self.topics, self.lda_model = self.train_lda(self.dict, self.corpus)
        print("ROUND 3 - Training LDA ended")
        self.trend_find()

    def preprocess(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs_2015, dfs_2016, dfs_2017 = utils.get_dataframe_from_json_list_by_year(jsons)
        # For fast check !
        dfs_2015 = dfs_2015[:1000]
        stop_words = stopwords.words('english')
        tokenized_doc = dfs_2015["body"].apply(lambda x: x.split())
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
        return dfs_2015, tokenized_doc

    def encoding(self, tokenized_doc):
        dictionary = corpora.Dictionary(tokenized_doc)
        corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
        return dictionary, corpus

    def train_lda(self, dict, corpus):
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.num_trends, id2word=dict, passes=15)
        topics = lda_model.print_topics(num_words=10)
        # for t in topics:
        #     print(t)
        return topics, lda_model

    def trend_find(self):
        topic_dict = {}
        print("TOPIC DICT IS BUILDING")
        for i, topic_list in enumerate(self.lda_model[self.corpus]):
            if not topic_list[0][0] in topic_dict:
                topic_dict[topic_list[0][0]] = [i]
            else:
                temp_list = topic_dict[topic_list[0][0]]
                temp_list.append(i)
                topic_dict[topic_list[0][0]] = temp_list
        print("TOPIC DICT HAS BEEN BUILT")
        for k in topic_dict:
            # print(topic_dict[k])
            print(self.common_phrase(self.tokenized_doc[topic_dict[k]], 2))
        return topic_dict

    def common_phrase(self, text_list, phrase_length):
        a = time.time()
        phrase_dict = {}
        for t in text_list:
            ngram = ngrams(t, phrase_length)
            for n in ngram:
                if n not in phrase_dict:
                    phrase_dict[n] = 0
                for t_t in text_list:
                    for i in range(len(t_t)-phrase_length+1):
                        if t_t[i] == n[0] and t_t[i+1] == n[1]:
                            phrase_dict[n] += 1
        result = sorted(phrase_dict.items(), key=lambda x: x[1], reverse=True)
        b = time.time()
        print(b-a)
        return result[:5]


if __name__ == "__main__":
    lda_class = LDA("../dataset", 20)
