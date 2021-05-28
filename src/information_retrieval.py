import math
import copy
from collections import Counter

import numpy as np

import preprocess

class OnIssueEventTracking:
    def __init__(self, df, issue_list, config):
        self.df = df
        self.issue_list = issue_list
        self.config = config

        self.preprocessor = preprocess.Preprocessor()
        self.detailed_info_extractor = DetailedInfoExtractor(df)
        self.information_retriever = InformationRetrieval(self.body_bow_list)

        self.cluster_number_to_docs = {cluster_number : self.get_cluster_number_to_docs(cluster_number) for cluster_number in set(df["cluster_number"].tolist())}
        self.cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number) for cluster_number in set(df["cluster_number"].tolist())}
        self.cluster_number_to_docs_and_avg_bow = sorted(self.cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number

        self.body_bow_list = [avg_bow for cluster_number, avg_bow in self.cluster_number_to_docs_and_avg_bow]
        self.idx_to_cluster_number = [cluster_number for cluster_number, avg_bow in self.cluster_number_to_docs_and_avg_bow]
        self.issue_bow_list = [self.get_bow_from_words(self.preprocessor.preprocess(issue)) for issue in issue_list]

    def apply_on_issue_event_tracking(self):
        for i, issue in enumerate(self.issue_bow_list):
            on_issue_event_clusters = self.on_issue_event_tracking(issue, self.body_bow_list, mode = self.config["on_issue_event_tracking"]["mode"])
            detailed_info = self.get_detailed_info_dict_from_event_clusters(on_issue_event_clusters)
            self.print_on_issue_event_tracking_result(self.issue_list[i], detailed_info.keys(), detailed_info)

    def get_total_bow(self, body_bow_list):
        total_bow = dict()
        for body_bow in body_bow_list:
            total_bow = dict(Counter(total_bow) + Counter(body_bow))
        return total_bow

    def get_bow_from_words(self, words):
        bow = dict()
        for word in words:
            if bow.get(word):
                bow[word] += 1
            else:
                bow[word] = 1
        return bow

    def get_cluster_number_to_docs(self, cluster_number):
        return self.df[self.df["cluster_number"] == cluster_number]["id"].tolist()

    def get_cluster_number_to_average_bow(self, cluster_number):
        cluster_bow_list = []
        for doc_id in self.cluster_number_to_docs[cluster_number]:
            cluster_bow_list.append(self.get_bow_from_words(self.preprocessor.preprocess(self.df["body"][doc_id])))
        total_cluster_bow = {k : (v / len(self.cluster_number_to_docs[cluster_number])) for k, v in self.get_total_bow(cluster_bow_list).items()}
        return total_cluster_bow

    def on_issue_event_tracking(self, issue, body_bow_list, mode = "normal", num_events = 5, weight_on_original_issue = 0.75): # mode = "normal" or "consecutive"
        on_issue_events = [] # index of each cluster

        if mode == "normal":
            body_score_list = []
            for body_bow in body_bow_list:
                body_score_list.append(self.information_retriever.score_document(issue, body_bow))
            body_score_dict = {k : v for k, v in enumerate(body_score_list)}

            on_issue_events = sorted(body_score_dict.items(), key = lambda x: x[1], reverse = True)[:num_events]
            on_issue_events = [on_issue_event[0] for on_issue_event in on_issue_events]
            on_issue_events = sorted(on_issue_events, key = lambda idx: self.df["time"][idx])

        else: # mode == "consecutive"
            while len(on_issue_events) < num_events:
                if len(on_issue_events) == 0:
                    in_order_issue = issue
                else:
                    original_issue = copy.deepcopy(issue)
                    temporary_issue = copy.deepcopy(self.body_bow_list[on_issue_events[-1]])
                    for key in original_issue:
                        original_issue[key] *= weight_on_original_issue
                    for key in temporary_issue:
                        temporary_issue[key] *= (1 - weight_on_original_issue)
                    in_order_issue = dict(Counter(original_issue) + Counter(temporary_issue))

                body_score_list = []
                for body_bow in body_bow_list:
                    body_score_list.append(self.information_retriever.score_document(in_order_issue, body_bow))
                body_score_dict = {k : v for k, v in enumerate(body_score_list)}

                on_issue_event_candidates = sorted(body_score_dict.items(), key = lambda x: x[1], reverse = True)[:num_events]
                on_issue_event_candidates = [on_issue_event_candidate[0] for on_issue_event_candidate in on_issue_event_candidates]

                in_order_event = None
                for candidate_event in on_issue_event_candidates:
                    if len(on_issue_events) == 0 or self.df["time"][candidate_event] > self.df["time"][on_issue_events[-1]]:
                        if in_order_event == None or self.df["time"][candidate_event] < self.df["time"][in_order_event]:
                            in_order_event = candidate_event

                if in_order_event == None:
                    break
                else:
                    on_issue_events.append(in_order_event)
        return on_issue_events

    def get_detailed_info_dict_from_event_clusters(self, event_clusters):
        # TODO: need to change self.df[idx, 0] (title) to event
        # TODO: extract person, organization, place from document using NER or something

        for idx in event_clusters:
            docs = self.cluster_number_to_docs[self.idx_to_cluster_number[idx]]
            print(docs)

        detailed_info = dict()
        for idx in event_clusters:
            detailed_info[self.df["title"][idx]] = dict({"person" : [], "organization" : [], "place" : []})
        return detailed_info

    def print_on_issue_event_tracking_result(self, issue, events, detailed_info):
        events_str = " -> ".join(events)
        cmd = f"[ Issue ]\n\n{issue}\n\n[ On-Issue Events ]\n\n{events_str}\n\n[ Detailed Information (per event) ]\n\n"
        for event, info in detailed_info.items():
            person = ", ".join(info["person"])
            organization = ", ".join(info["organization"])
            place = ", ".join(info["place"])

            detailed_info_str = f"Event: {event}\n\n"
            detailed_info_str += f"    -    Person: {person}\n"
            detailed_info_str += f"    -    Organizaiton: {organization}\n"
            detailed_info_str += f"    -    Place: {place}\n\n"
            cmd += detailed_info_str
        print(cmd)

class DetailedInfoExtractor:
    def __init__(self, df):
        self.df = df

    def get_event_summary_from_doc_id(doc_id):
        return self.df["title"][doc_id]

    def get_person_list_from_doc_id(doc_id):
        return []

    def get_organization_list_from_doc_id(doc_id):
        return []

    def get_place_list_from_doc_id(doc_id):
        return []

class InformationRetrieval:
    def __init__(self, body_bow_list):
        self.body_bow_list = body_bow_list

    def score_document(self, query, document, mode = "tfidf"): # mode = "tfidf" or "bm25"
        if self.norm(query) * self.norm(document) == 0:
            return 0

        score = 0
        terms = set(query.keys()) & set(document.keys())
        for term in terms:
            if mode == "tfidf":
                score += self.tfidf(term, query) * self.tfidf(term, document)
            else: # "bm25"
                score += self.bm25(term, query) * self.bm25(term, document)
        score /= self.norm(query) * self.norm(document)
        return score

    def bm25(self, term, document):
        k1, b, N, n = 2.0, 0.75, len(self.body_bow_list), 0
        for body_bow in self.body_bow_list:
            if body_bow.get(term):
                n += 1

        doc_length = np.sum(document.values())
        avg_doc_length = np.mean([np.sum(body_bow.values()) for body_bow in self.body_bow_list])

        denom = k1 * ((1- b) + (b * doc_length / avg_doc_length)) + self.tf(term, document)
        return self.tf(term, document) / denom * math.log(N - n + 0.5 / (n + 0.5))

    def tfidf(self, term, document):
        return self.tf(term, document) * self.idf(term, document)
    
    def tf(self, term, document, mode = "l"):
        if document[term] == 0:
            return 0

        if mode == "n":
            return document[term]
        elif mode == "l":
            return 1 + math.log(document[term])
        elif mode == "a":
            return 0.5 + (0.5 * document[term]) / max(document.values())
        else: # log ave
            return (1 + math.log(document[term])) / (1 + math.log(sum(document.values()) / len(document.values())))

    def idf(self, term, document, mode = "t"):
        if document[term] == 0:
            return 0

        df = 0 # document frequency
        for body_bow in self.body_bow_list:
            if body_bow.get(term):
                df += 1

        if mode == "n":
            return 1
        elif mode == "t":
            return math.log(len(self.body_bow_list) / df)
        else: # prob idf
            return max(0, math.log((len(self.body_bow_list) - df / df)))

    def norm(self, document, mode = "c"):
        if mode == "n":
            return 1
        elif mode == "c":
            return np.linalg.norm(list(document.values()))