import preprocess
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import spacy
from spacy import displacy
import en_core_web_sm


def person_replace(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    person_list = []
    for entity in doc.ents:
        # print(f'{entity.text:12} \t {entity.label_}')
        if entity.label_ == 'PERSON':
            person_list.append(entity.text)
    news_str = str(doc)
    for p in person_list:
        news_str = news_str.replace(p, "person")
    return news_str

class LSA():
    def __init__(self, data_dir, num_trends):
        self.num_trends = num_trends
        self.dfs, self.tokenized_doc = self.pre_process(data_dir)
        print("Preprocess ended")
        self.tfidf = self.tfidf_LSA(self.dfs)

    def pre_process(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs_2015, dfs_2016, dfs_2017 = utils.get_dataframe_from_json_list_by_year(jsons)
        dfs_2015 = dfs_2015[:200]
        Processor = preprocess.Preprocessor()
        dfs_2015["body"] = dfs_2015["body"].apply(lambda x: person_replace(x))
        dfs_2015["body"] = dfs_2015["body"].apply(lambda x: Processor.preprocess(x))
        dfs_2015["body"] = dfs_2015["body"].apply(lambda x: ' '.join([word for word in x if len(word) > 3]))
        tokenized_doc = dfs_2015["body"].apply(lambda x: x.split())
        return dfs_2015, tokenized_doc

    def tfidf_LSA(self, dfs):
        vectorizer = TfidfVectorizer(stop_words='english',
                                     max_features=1000,
                                     max_df=0.5,
                                     smooth_idf=True)
        X = vectorizer.fit_transform(dfs["body"])
        svd_model = TruncatedSVD(n_components=self.num_trends, algorithm='randomized',
                                 n_iter=100, random_state=424)
        svd_model.fit(X)
        terms = vectorizer.get_feature_names()  # 단어 집합. 1,000개의 단어가 저장됨.
        n = 5
        for idx, topic in enumerate(svd_model.components_):
            print("Topic %d:" % (idx + 1),
                    [(terms[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])


if __name__ == "__main__":
    lsa = LSA("../dataset", 20)