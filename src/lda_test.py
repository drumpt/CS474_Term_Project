import gensim
from gensim import corpora
from nltk.corpus import stopwords
import utils


class LDA:
    def __init__(self, data_dir, num_trends):
        # self.data_dir = data_dir
        self.num_trends = num_trends
        self.tokenized_doc = self.preprocess(data_dir)
        self.dict, self.corpus = self.encoding(self.tokenized_doc)
        self.topics = self.train_lda(self.dict, self.corpus)
        for topic in self.topics:
            print(topic)

    def preprocess(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs = utils.get_dataframe_from_json_list(jsons)
        stop_words = stopwords.words('english')
        tokenized_doc = dfs["body"].apply(lambda x: x.split())
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
        return tokenized_doc

    def encoding(self, tokenized_doc):
        dictionary = corpora.Dictionary(tokenized_doc)
        corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
        # print(corpus[1])
        return dictionary, corpus

    def train_lda(self, dict, corpus):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.num_trends, id2word=dict, passes=15)
        topics = ldamodel.print_topics(num_words=5)
        return topics

    def get_topics(self):
        return self.topics

if __name__ == "__main__":
    lda_class = LDA("../dataset", 10)