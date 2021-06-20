import enum
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import math
import random
import time
from datetime import datetime
import copy
from collections import Counter

import nltk
nltk.download('averaged_perceptron_tagger')
import numpy as np
from transformers import pipeline
from submodules.bert_ner.bert import Ner
from nltk.tag import StanfordNERTagger, pos_tag
#from gensim.summarization.summarizer import summarize

import preprocess


class OnIssueEventTracking:
    def __init__(self, df, issue_list, config):
        self.df = df
        self.issue_list = issue_list
        self.config = config

        print("1")

        self.preprocessor = preprocess.Preprocessor()
        self.detailed_info_extractor = DetailedInfoExtractor(df, config)

        print("2")

        self.cluster_number_to_docs = {cluster_number : self.get_cluster_number_to_docs(cluster_number) for cluster_number in set(df["cluster_number"].tolist())}
        self.cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number) for cluster_number in set(df["cluster_number"].tolist())}
        self.cluster_number_to_docs_and_avg_bow = sorted(self.cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number

        print("3")

        self.body_bow_list = [avg_bow for cluster_number, avg_bow in self.cluster_number_to_docs_and_avg_bow]
        self.idx_to_cluster_number = [cluster_number for cluster_number, avg_bow in self.cluster_number_to_docs_and_avg_bow]
        self.issue_bow_list = [self.get_bow_from_words(self.preprocessor.preprocess(issue)) for issue in issue_list]

        print("4")

        self.information_retriever = InformationRetrieval(self.body_bow_list)

    def apply_on_issue_event_tracking(self):
        for i, issue in enumerate(self.issue_bow_list):
            on_issue_event_clusters = self.on_issue_event_tracking(issue, self.body_bow_list, method = self.config["on_issue_event_tracking"]["method"])
            detailed_info = self.get_detailed_info_list_from_event_clusters(on_issue_event_clusters)
            self.print_on_issue_event_tracking_result(self.issue_list[i], detailed_info)

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

    def on_issue_event_tracking(self, issue, body_bow_list, method = "normal", num_events = 5, 
    weight_on_original_issue = 0.8): # method = "normal" or "consecutive"
        def time_to_timestamp(t):
            return time.mktime(datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timetuple())

        def get_average_timestamp_from_cluster_number(cluster_number):
            result = 0
            docs = self.cluster_number_to_docs[self.idx_to_cluster_number[cluster_number]]
            for doc_id in docs:
                result += time_to_timestamp(self.df["time"][doc_id])
            return result / len(docs)

        on_issue_events = [] # index of each cluster

        if method == "normal":
            body_score_list = []
            for body_bow in body_bow_list:
                body_score_list.append(self.information_retriever.score_document(issue, body_bow))
            body_score_dict = {k : v for k, v in enumerate(body_score_list)}

            on_issue_events = sorted(body_score_dict.items(), key = lambda x: x[1], reverse = True)[:num_events]
            on_issue_events = [on_issue_event[0] for on_issue_event in on_issue_events]
            on_issue_events = sorted(on_issue_events, key = lambda idx: get_average_timestamp_from_cluster_number(idx))

        else: # method == "consecutive"
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
                    if len(on_issue_events) == 0 or get_average_timestamp_from_cluster_number(candidate_event) > get_average_timestamp_from_cluster_number(on_issue_events[-1]):
                        if in_order_event == None or get_average_timestamp_from_cluster_number(candidate_event) < get_average_timestamp_from_cluster_number(in_order_event):
                            in_order_event = candidate_event

                if in_order_event == None:
                    break
                else:
                    on_issue_events.append(in_order_event)
        return on_issue_events

    def get_detailed_info_list_from_event_clusters(self, event_clusters):
        detailed_info_list = []

        for idx in event_clusters:
            event_summary, person_list, organization_list, place_list = "", [], [], []

            docs = self.cluster_number_to_docs[self.idx_to_cluster_number[idx]]
            random_doc = random.choice(docs) # TODO: consider another method

            # extract detailed information
            event_summary = self.detailed_info_extractor.get_event_summary_from_doc_id(random_doc)
            for doc_id in docs:
                temp_person_list, temp_organization_list, temp_place_list = self.detailed_info_extractor.get_detailed_info_list_from_doc_id(doc_id)

                person_list.extend(temp_person_list)
                organization_list.extend(temp_organization_list)
                place_list.extend(temp_place_list)

            # remove redundant
            person_list = list(set(person_list))
            organization_list = list(set(organization_list))
            place_list = list(set(place_list))

            detailed_info_list.append((event_summary, person_list, organization_list, place_list))
        return detailed_info_list

    def print_on_issue_event_tracking_result(self, issue, detailed_info_list):
        events_str = " -> ".join([detailed_info[0] for detailed_info in detailed_info_list])
        total_str = f"[ Issue ]\n\n{issue}\n\n[ On-Issue Events ]\n\n{events_str}\n\n[ Detailed Information (per event) ]\n\n"

        for event_summary, person_list, organization_list, place_list in detailed_info_list:
            person_str = ", ".join(person_list)
            organization_str = ", ".join(organization_list)
            place_str = ", ".join(place_list)

            detailed_info_str = f"Event: {event_summary}\n\n"
            detailed_info_str += f"    -    Person: {person_str}\n"
            detailed_info_str += f"    -    Organizaiton: {organization_str}\n"
            detailed_info_str += f"    -    Place: {place_str}\n\n"
            total_str += detailed_info_str
        print(total_str)


class RelatedIssueEventTracking:
    def __init__(self, df, inverted_index, issue_list, config):
        self.df = df
        self.inverted_index = inverted_index
        self.issue_list = issue_list # used instead of issue_bow_list in apply_related_issue_event_tracking
        self.config = config

        self.preprocessor = preprocess.Preprocessor()
        self.detailed_info_extractor = DetailedInfoExtractor(df, config)

        self.cluster_number_to_docs = {cluster_number : self.get_cluster_number_to_docs(cluster_number) for cluster_number in set(df["cluster_number"].tolist())}
        self.cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number) for cluster_number in set(df["cluster_number"].tolist())}
        self.cluster_number_to_docs_and_avg_bow = sorted(self.cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number

        self.body_bow_list = [avg_bow for cluster_number, avg_bow in self.cluster_number_to_docs_and_avg_bow]
        self.idx_to_cluster_number = [cluster_number for cluster_number, avg_bow in self.cluster_number_to_docs_and_avg_bow]

        """self.information_retriever = InformationRetrieval(self.body_bow_list)"""
        self.information_retriever = InformationRetrieval(self.df, inverted_index, config)

    def apply_related_issue_event_tracking(self):
        def get_candidate_cluster_numbers(issue):
            candidate_cluster_numbers = set()
            for issue_token in issue:
                for doc_id in self.inverted_index.get(issue_token):
                    for cluster_number in self.df[self.df["id"] == doc_id]["cluster_number"].tolist():
                        candidate_cluster_numbers.add(cluster_number)
            return candidate_cluster_numbers
            
        for i, issue in enumerate(self.issue_list):
            issue_words = self.preprocessor.preprocess(issue)
            candidate_cluster_numbers = get_candidate_cluster_numbers(issue_words)

            ngram_body_bow_list = []
            ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=len(issue_words)) for cluster_number in candidate_cluster_numbers}
            ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
            ngram_body_bow_list.append({cluster_number: avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow})
            if (len(issue_words) == 1):
                ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=1) for cluster_number in candidate_cluster_numbers}
                ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
                ngram_body_bow_list.append({cluster_number: avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow})
            else:
                ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=len(issue_words) // 2) for cluster_number in candidate_cluster_numbers}
                ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
                ngram_body_bow_list.append({cluster_number: avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow})
                
                if (len(issue_words) % 2):
                    ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=len(issue_words) - len(issue_words) // 2) for cluster_number in candidate_cluster_numbers}
                    ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
                    ngram_body_bow_list.append({cluster_number: avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow})
                else:
                    ngram_body_bow_list.append(ngram_body_bow_list[len(ngram_body_bow_list) - 1])

            related_issue_event_clusters = self.related_issue_event_tracking(issue_words, self.inverted_index, ngram_body_bow_list)
            detailed_info = self.get_detailed_info_list_from_event_clusters(related_issue_event_clusters)
            self.print_related_issue_event_tracking_result(issue, detailed_info)
        """df = self.df
        for i, issue in enumerate(self.issue_list):
            issue_words = self.preprocessor.preprocess(issue)

            ngram_body_bow_list = []
            ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=len(issue_words)) for cluster_number in set(df["cluster_number"].tolist())}
            ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
            ngram_body_bow_list.append([avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow])
            if (len(issue_words) == 1):
                ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=1) for cluster_number in set(df["cluster_number"].tolist())}
                ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
                ngram_body_bow_list.append([avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow])
            else:
                ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=len(issue_words) // 2) for cluster_number in set(df["cluster_number"].tolist())}
                ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
                ngram_body_bow_list.append([avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow])
                ngram_cluster_number_to_avg_bow = {cluster_number : self.get_cluster_number_to_average_bow(cluster_number, ngram=len(issue_words) - len(issue_words) // 2) for cluster_number in set(df["cluster_number"].tolist())}
                ngram_cluster_number_to_docs_and_avg_bow = sorted(ngram_cluster_number_to_avg_bow.items(), key = lambda item: item[0]) # sorted by clsuter_number
                ngram_body_bow_list.append([avg_bow for cluster_number, avg_bow in ngram_cluster_number_to_docs_and_avg_bow])

            related_issue_event_clusters = self.related_issue_event_tracking(issue_words, ngram_body_bow_list)
            detailed_info = self.get_detailed_info_list_from_event_clusters(related_issue_event_clusters)
            self.print_related_issue_event_tracking_result(issue, detailed_info)"""

    def get_total_bow(self, body_bow_list):
        total_bow = dict()
        for body_bow in body_bow_list:
            total_bow = dict(Counter(total_bow) + Counter(body_bow))
        return total_bow

    def get_bow_from_words(self, words, n = 1):
        bow = dict()
        for word in words:
            if bow.get(word):
                bow[word] += 1
            else:
                bow[word] = 1
        return bow

    def get_ngram_bow_from_words(self, words, n = 1):
        bow = dict()
        for i in range (len(words) - n + 1):
            ngram = ()
            for j in range (n):
                ngram += (words[i + j],)
            if bow.get(ngram):
                bow[ngram] += 1
            else:
                bow[ngram] = 1
        return bow

    def get_cluster_number_to_docs(self, cluster_number):
        return self.df[self.df["cluster_number"] == cluster_number]["id"].tolist()

    def get_cluster_number_to_average_bow(self, cluster_number, ngram=0):
        if ngram:
            cluster_bow_list = []
            for doc_id in self.cluster_number_to_docs[cluster_number]:
                cluster_bow_list.append(self.get_ngram_bow_from_words(self.preprocessor.preprocess(self.df["body"][doc_id]), n=ngram))
            total_cluster_bow = {k : (v / len(self.cluster_number_to_docs[cluster_number])) for k, v in self.get_total_bow(cluster_bow_list).items()}
            return total_cluster_bow
        else:
            cluster_bow_list = []
            for doc_id in self.cluster_number_to_docs[cluster_number]:
                cluster_bow_list.append(self.get_bow_from_words(self.preprocessor.preprocess(self.df["body"][doc_id])))
            total_cluster_bow = {k : (v / len(self.cluster_number_to_docs[cluster_number])) for k, v in self.get_total_bow(cluster_bow_list).items()}
            return total_cluster_bow

    def related_issue_event_tracking(self, issue, inverted_index, cluster_to_body_bow_dict_list, num_events = 4):
        related_issue_events = [] # index of each cluster

        body_score_list = []
        issue_bow_subset_list = []
        issue_total_bow = self.get_ngram_bow_from_words(issue, n = len(issue))

        if len(issue) == 1:
            issue_bow_subset_list.append(self.get_ngram_bow_from_words(issue, n=1))
        else:
            """if len(issue) % 2:
                issue_bow_subset_list.append(self.get_bow_from_words(issue[:len(issue) // 2]))
                issue_bow_subset_list.append(self.get_bow_from_words(issue[len(issue) // 2:]))
                #issue_bow_subset_list.append(self.get_bow_from_words(issue[:len(issue) // 2 + 1]))
                #issue_bow_subset_list.append(self.get_bow_from_words(issue[len(issue) // 2 + 1:]))
            else:
                issue_bow_subset_list.append(self.get_bow_from_words(issue[:len(issue) // 2]))
                issue_bow_subset_list.append(self.get_bow_from_words(issue[len(issue) // 2:]))"""
            issue_bow_subset_list.append(self.get_ngram_bow_from_words(issue[:len(issue) // 2], n=len(issue) // 2))
            issue_bow_subset_list.append(self.get_ngram_bow_from_words(issue[len(issue) // 2:], n=len(issue) - len(issue) // 2))

        related_issue_body_score_dict = dict()
        for i, issue_bow in enumerate(issue_bow_subset_list):
            for j, body_bow in cluster_to_body_bow_dict_list[i + 1].items():
                exists2 = True
                for item in issue_bow.items():
                    if not item[0] in body_bow:
                        exists2 = False
                        break
                if exists2:
                    exists1 = False
                    for item in issue_total_bow.items():
                        if item[0] in cluster_to_body_bow_dict_list[0][j]:
                            exists1 = True
                            break
                    if not exists1:
                        score = self.information_retriever.score_document(issue_bow, body_bow, ngram_body_bow_list=cluster_to_body_bow_dict_list[i + 1])
                        if j in related_issue_body_score_dict:
                            if score > related_issue_body_score_dict[j]:
                                related_issue_body_score_dict[j] = score
                        else:
                            related_issue_body_score_dict[j] = score
            """related_issue_body_score_list = []
            for body_bow in body_bow_list[0]:
                exists1 = False
                for item in issue_total_bow.items():
                    if item[0] in body_bow:
                        exists1 = True
                exists2 = True
                for item in issue_bow.items():
                    if not item[0] in body_bow:
                        exists2 = False
                if exists2 and not exists1:
                    related_issue_body_score_list.append(self.information_retriever.score_document(issue_bow, body_bow, ngram_body_bow_list=body_bow_list[i + 1]))
                else:
                    related_issue_body_score_list.append(0)
            for k, v in enumerate(related_issue_body_score_list):
                if k in related_issue_body_score_dict:
                    if related_issue_body_score_dict[k] > v:
                        value = related_issue_body_score_dict[k]
                    else:
                        value = v
                else:
                    value = v * 1.2
                related_issue_body_score_dict[k] = value"""

        related_issue_events = sorted(related_issue_body_score_dict.items(), key = lambda x: x[1], reverse = True)[:num_events]
        related_issue_events = [related_issue_event[0] for related_issue_event in related_issue_events]
        
        return related_issue_events

    def get_detailed_info_list_from_event_clusters(self, event_clusters):
        detailed_info_list = []

        for idx in event_clusters:
            event_summary, person_list, organization_list, place_list = "", [], [], []

            docs = self.cluster_number_to_docs[self.idx_to_cluster_number[idx]]
            random_doc = random.choice(docs) # TODO: consider another method

            # extract detailed information
            event_summary = self.detailed_info_extractor.get_event_summary_from_doc_id(random_doc)
            for doc_id in docs:
                temp_person_list, temp_organization_list, temp_place_list = self.detailed_info_extractor.get_detailed_info_list_from_doc_id(doc_id)

                person_list.extend(temp_person_list)
                organization_list.extend(temp_organization_list)
                place_list.extend(temp_place_list)

            # remove redundant
            person_list = list(set(person_list))
            organization_list = list(set(organization_list))
            place_list = list(set(place_list))

            detailed_info_list.append((event_summary, person_list, organization_list, place_list))
        return detailed_info_list

    def print_related_issue_event_tracking_result(self, issue, detailed_info_list):
        events_str = ", ".join([detailed_info[0] for detailed_info in detailed_info_list])
        total_str = f"[ Issue ]\n\n{issue}\n\n[ Related-Issue Events ]\n\n{events_str}\n\n[ Detailed Information (per event) ]\n\n"

        for event_summary, person_list, organization_list, place_list in detailed_info_list:
            person_str = ", ".join(person_list)
            organization_str = ", ".join(organization_list)
            place_str = ", ".join(place_list)

            detailed_info_str = f"    Event: {event_summary}\n\n"
            detailed_info_str += f"        -    Person: {person_str}\n"
            detailed_info_str += f"        -    Organizaiton: {organization_str}\n"
            detailed_info_str += f"        -    Place: {place_str}\n\n"
            total_str += detailed_info_str
        print(total_str)


class DetailedInfoExtractor:
    def __init__(self, df, config):
        self.df = df

        # Text Summarization
        self.summary_method = config["detailed_info_extractor"]["summary_method"]
        self.summary_target = config["detailed_info_extractor"]["summary_target"]
        if self.summary_method == "transformer": # transformer, textrank
            self.summarizer = pipeline('summarization')
        elif self.summary_method == "textrank":
            self.summarizer = summarize

        # NER
        self.ner_method = config["detailed_info_extractor"]["ner_method"]
        if self.ner_method == "bert":
            self.named_entity_recognizer = Ner(config["detailed_info_extractor"]["ner_model_dir"])
            self.label_map = {
                "1": "O", # None
                "2": "B-MISC", # Begin of Artifact, Event, Natural Phenomenon
                "3": "I-MISC", # Inside of Artifact, Event, Natural Phenomenon
                "4": "B-PER", # Begin of Person
                "5": "I-PER", # Inside of Person
                "6": "B-ORG", # Begin of Organization
                "7": "I-ORG", # Inside of Organization
                "8": "B-LOC", # Begin of Geographical Entity
                "9": "I-LOC", # Inside of Geographical Entity
                "10": "[CLS]", # Special Classifcation Token
                "11": "[SEP]" # Sentencec Pair Token
            }
        elif self.ner_method == "stanford":
            os.environ["CLASSPATH"] = "models"
            os.environ["JAVAHOME"] = "/usr/bin/java"
            self.named_entity_recognizer = StanfordNERTagger("models/stanford-ner.jar")
        self.proper_noun_tagger = pos_tag

    def get_event_summary_from_doc_id(self, doc_id):
        ratio = {"title_summary": 0.8, "body_summary": 0.1, "description_summary": 0.3}

        if self.summary_target == "title":
            return self.df["title"][doc_id]
        elif self.summary_target == "title_summary":
            summary_target = self.df["title"][doc_id]
        elif self.summary_target == "body_summary":
            summary_target = self.df["body"][doc_id]
        elif self.summary_target == "description_summary":
            summary_target = self.df["description"][doc_id]

        if self.summary_method == "transformer":
            return self.summarizer(summary_target)[0]["summary_text"]
        elif self.summary_method == "textrank":
            return self.summarizer(summary_target, ratio = ratio[self.summary_target])

    def get_detailed_info_list_from_doc_id(self, doc_id):
        person_list, organization_list, place_list = [], [], []

        if self.ner_method == "bert":
            token_position = 0
            for sentence in self.df["body"][doc_id].split(".")[:-1]: # remove last sentence regarding newspaper company
                for ner_dict in self.named_entity_recognizer.predict(sentence):
                    print(ner_dict)
                    if ner_dict['tag'] in ["B-PER", "I-PER"]:
                        person_list.append((token_position, ner_dict['tag'], ner_dict['word']))
                    elif ner_dict['tag'] in ["B-ORG", "I-ORG"]:
                        organization_list.append((token_position, ner_dict['tag'], ner_dict['word']))
                    elif ner_dict['tag'] in ["B-LOC", "I-LOC"]:
                        place_list.append((token_position, ner_dict['tag'], ner_dict['word']))
                    token_position += 1
            return self.parse_detailed_info(person_list, "person"), self.parse_detailed_info(organization_list, "organization"), self.parse_detailed_info(place_list, "place")

        elif self.ner_method == "stanford":
            for token, tag in self.named_entity_recognizer.tag(self.df["body"][doc_id].split(".")):
                print(token)
                if tag == "PERSON":
                    person_list.append(token)
                elif tag == "ORGANIZATION":
                    organization_list.append(token)
                elif tag == "LOCATION":
                    place_list.append(token)
            return person_list, organization_list, place_list

    def parse_detailed_info(self, detailed_info_list, info_type):
        parsed_list = []
        info_type_to_tokens = {
            "person" : {"tag_start_token" : "B-PER", "tag_end_token" : "I-PER"},
            "organization" : {"tag_start_token" : "B-ORG", "tag_end_token" : "I-ORG"},
            "place" : {"tag_start_token" : "B-LOC", "tag_end_token" : "I-LOC"}
        }

        print(detailed_info_list)

        adjacency_idx = -2
        cumulative_word = ""
        for idx, tag, word in detailed_info_list:
            if tag == info_type_to_tokens[info_type]["tag_start_token"]:
                if cumulative_word != "":
                    parsed_list.append(cumulative_word)
                cumulative_word = word
                adjacency_idx = idx
            elif tag == info_type_to_tokens[info_type]["tag_end_token"] and idx == adjacency_idx + 1:
                cumulative_word += " " + word
                adjacency_idx = idx
            elif tag == info_type_to_tokens[info_type]["tag_end_token"] and idx != adjacency_idx + 1:
                if cumulative_word != "":
                    parsed_list.append(cumulative_word)
                cumulative_word = word
        if cumulative_word != "":
            parsed_list.append(cumulative_word)

        return self.postprocess_detailed_info(parsed_list)

    def postprocess_detailed_info(self, parsed_list):
        duplicated_words, nonproper_words = [], []

        print("\n\n\n\n\n")
        print("postprocess_detailed_info")
        print(parsed_list)
        print("\n\n\n\n\n")

        for inner_word in parsed_list:
            for outer_word in parsed_list:
                if inner_word.lower() != outer_word.lower() and inner_word.lower() in outer_word.lower():
                    duplicated_words.append(inner_word)

        for word, pos in self.proper_noun_tagger(parsed_list): # consider only proper nouns(NNP, NNPS)
            print(word, pos)
            if not pos in ['NN', 'NNP', 'NNPS']: # TODO: remove NN
                nonproper_words.append(word)

        return list(set([word for word in parsed_list if word not in duplicated_words and word not in nonproper_words]))


class InformationRetrieval:
    """def __init__(self, body_bow_list):
        self.body_bow_list = body_bow_list"""
    def __init__(self, df, inverted_index, config):
        self.df = df
        self.inverted_index = inverted_index
        self.body_bow_list = [self.get_bow_from_words(words)  for _, words in self.df["preprocessed_body"].iteritems()]
        self.total_bow =  self.get_total_bow(self.body_bow_list)
        self.sorted_words = sorted(self.total_bow.keys())
        self.output_dir = config["vectorize"]["final_dataframe_dir"]

    def get_bow_from_words(self, words):
        bow = dict()
        for word in words:
            if bow.get(word):
                bow[word] += 1
            else:
                bow[word] = 1
        return bow

    def get_total_bow(self, body_bow_list):
        total_bow = dict()
        for body_bow in body_bow_list:
            total_bow = dict(Counter(total_bow) + Counter(body_bow))
        return total_bow

    def get_tfidf_vector_list(self, mode = "bm25"):
        tfidf_vector_list = []
        vector_length = len(self.sorted_words)

        for body_bow in self.body_bow_list:
            tfidf_vector = np.zeros(vector_length)
            for word in body_bow.keys():
                if mode == "tfidf":
                    tfidf_vector[self.sorted_words.index(word)] = self.tfidf(word, body_bow, self.inverted_index)
                else: # "bm25"
                    tfidf_vector[self.sorted_words.index(word)] = self.bm25(word, body_bow, self.inverted_index)
            tfidf_vector_list.append(tfidf_vector)

        self.df["tfidf_vector"] = tfidf_vector_list
        with open(self.output_dir, "wb") as f:
            pickle.dump(self.df, f)
        return self.df

    def score_document(self, query, document, mode = "tfidf", ngram_body_bow_list=None): # mode = "tfidf" or "bm25"
        if self.norm(query) * self.norm(document) == 0:
            return 0

        self.ngram_body_bow_list = ngram_body_bow_list

        score = 0
        terms = set(query.keys()) & set(document.keys())
        for term in terms:
            if mode == "tfidf":
                score += self.tfidf(term, query) * self.tfidf(term, document)
            else: # "bm25"
                score += self.bm25(term, query) * self.bm25(term, document)
        score /= self.norm(query) * self.norm(document)
        return score

    """def bm25(self, term, document):
        k1, b, N, n = 2.0, 0.75, len(self.body_bow_list), 0
        for body_bow in self.body_bow_list:
            if body_bow.get(term):
                n += 1

        doc_length = np.sum(document.values())
        avg_doc_length = np.mean([np.sum(body_bow.values()) for body_bow in self.body_bow_list])

        denom = k1 * ((1- b) + (b * doc_length / avg_doc_length)) + self.tf(term, document)
        return self.tf(term, document) / denom * math.log(N - n + 0.5 / (n + 0.5))"""

    def bm25(self, term, document, inverted_index):
        k1, b, N, n = 2.0, 0.75, len(self.body_bow_list), len(inverted_index[term])

        doc_length = np.sum(list(document.values()))
        avg_doc_length = np.mean([np.sum(list(body_bow.values())) for body_bow in self.body_bow_list])

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

    def idf(self, term, document, inverted_index, mode = "t"):
        if document[term] == 0:
            return 0

        """df = 0 # document frequency
        if self.ngram_body_bow_list:
            for body_bow in self.ngram_body_bow_list:
                if body_bow.get(term):
                    df += 1
        else:
            for body_bow in self.body_bow_list:
                if body_bow.get(term):
                    df += 1"""
        if self.ngram_body_bow_list:
            df = 0
            for body_bow in self.ngram_body_bow_list:
                if body_bow.get(term):
                    df += 1
        else:
            df = len(inverted_index[term])

        if mode == "n":
            return 1
        elif mode == "t":
            if self.ngram_body_bow_list:
                return math.log(len(self.ngram_body_bow_list) / df)
            return math.log(len(self.body_bow_list) / df)
        else: # prob idf
            if self.ngram_body_bow_list:
                return max(0, math.log((len(self.ngram_body_bow_list) - df / df)))
            return max(0, math.log((len(self.body_bow_list) - df / df)))

    def norm(self, document, mode = "c"):
        if mode == "n":
            return 1
        elif mode == "c":
            return np.linalg.norm(list(document.values()))