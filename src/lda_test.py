import gensim
import preprocess
from gensim import corpora
from nltk.corpus import stopwords
from nltk.util import ngrams
import utils
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDA:
    def __init__(self, data_dir, num_trends):
        self.num_trends = num_trends
        self.dfs, self.tokenized_doc = self.pre_process(data_dir)
        print("ROUND 1 - Preprocess ended")
        self.dict, self.corpus = self.encoding(self.tokenized_doc)
        print("ROUND 2 - Encoding ended")
        self.topics, self.lda_model = self.train_lda(self.dict, self.corpus)
        print("ROUND 3 - Training LDA ended")
        self.trend_find()

    def pre_process(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs_2015, dfs_2016, dfs_2017 = utils.get_dataframe_from_json_list_by_year(jsons)
        # For fast check !
        dfs_2015 = dfs_2015
        Processor = preprocess.Preprocessor()
        tokenized_doc = dfs_2015["body"].apply(lambda x: Processor.preprocess(x))
        # stop_words = stopwords.words('english')
        # tokenized_doc = dfs_2015["body"].apply(lambda x: x.split())
        # tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
        # print(f"tokenized doc is {tokenized_doc}")
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
        # test_list = []
        for i, topic_list in enumerate(self.lda_model[self.corpus]):
            # print("FOR CHECK")
            # test_list.append(topic_list[0][1])
            if topic_list[0][1] < 0.75:
                continue
            if not topic_list[0][0] in topic_dict:
                topic_dict[topic_list[0][0]] = [i]
            else:
                temp_list = topic_dict[topic_list[0][0]]
                temp_list.append(i)
                topic_dict[topic_list[0][0]] = temp_list
        print("TOPIC DICT HAS BEEN BUILT")

        # test_list.sort(reverse=True)
        # cnt = 0
        # standard = 0.9
        # for tl in test_list:
        #     if tl > standard:
        #         cnt += 1
        #     else:
        #         print(f"Above {standard} are {cnt}")
        #         cnt = 0
        #         standard -= 0.1

        for k in topic_dict:
            print(f"{k}th topic can be seen as below:")
            # bi = self.common_phrase(self.tokenized_doc[topic_dict[k]], 2)
            tri = self.common_phrase(self.tokenized_doc[topic_dict[k]], 3)
            # quad = self.common_phrase(self.tokenized_doc[topic_dict[k]], 4)
            result_arr = [tri[0]]
            result_arr.sort(key=lambda x:x[1], reverse=True)
            print(f"Best ngram was {result_arr[0]}")
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
        print(f"Function {phrase_length}-length common_phrase time : {b-a}")
        return result[:5]

    def get_topics(self):
        return self.topics


class LDA_scikit():
    def __init__(self, directory, num_trends):
        self.num_trends = num_trends
        self.dfs, self.tokenized_doc = self.pre_process(directory)
        print("Preprocess ended")
        self.detokenized = self.detokenization(self.tokenized_doc)
        print("Detokenization ended")
        self.tfidf = self.tfidf_LDA(self.detokenized)
        print("Title based LDA ended")

    def pre_process(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs_2015, dfs_2016, dfs_2017 = utils.get_dataframe_from_json_list_by_year(jsons)
        # dfs_2015 = dfs_2015
        Processor = preprocess.Preprocessor()
        tokenized_doc = dfs_2015["title"].apply(lambda x: Processor.preprocess(x))
        tokenized_doc = tokenized_doc.apply(lambda x: [word for word in x if len(word) > 3])
        return dfs_2015, tokenized_doc

    def detokenization(self, tokenized_doc):
        detokenized = []
        for i in range(len(tokenized_doc)):
            t = ' '.join(tokenized_doc[i])
            detokenized.append(t)
        return detokenized

    def tfidf_LDA(self, detokenized):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # 상위 1,000개의 단어를 보존
        X = vectorizer.fit_transform(detokenized)
        lda_model = LatentDirichletAllocation(n_components=self.num_trends, learning_method='online',
                                              random_state=777, max_iter=1)
        lda_top = lda_model.fit_transform(X)
        terms = vectorizer.get_feature_names()
        n = 5
        for idx, topic in enumerate(lda_model.components_):
            print("Topic %d:" % (idx + 1), [(terms[i], topic[i].round(2))
                                            for i in topic.argsort()[:-n - 1:-1]])
        # print(f"LDA top is {lda_top}")
        return 1


if __name__ == "__main__":
    # lda_class = LDA("../dataset", 10)
    lda_scikit = LDA_scikit("../dataset", 30)
