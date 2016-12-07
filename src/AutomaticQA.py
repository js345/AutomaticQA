'''
AutomaticQA AutomaticQA
Created on 11/30/16
@author: xiaofo
'''

from __future__ import division
from src.QueryLikelihoodModel import QueryLikelihoodModel
from src.LMHRank import LMHRANK
from src.WordTranslation import WordTranslation
from src.TranslationModel import TranslationModel
import json


class AutomaticQA:

    def __init__(self, data, mu=300, λ=0.5, threshold=0.05):
        self.data = data
        self.title = data['Title']
        self.body = data['Body']
        self.answer = data['AnswerBody']
        self.mu = mu
        self.λ = λ
        self.threshold = threshold
        self.queryLikelihoodModel = QueryLikelihoodModel(self.answer, mu)
        self.LMHranks = None
        self.translation_table = None
        self.translation_model = None

    def train(self):
        self.queryLikelihoodModel.build_model()
        self.LMHranks = LMHRANK.compute_scores(self.queryLikelihoodModel)
        self.translation_table = self.calculate_word_translation(self.threshold)
        self.translation_model = TranslationModel(self.answer, self.λ, self.translation_table,
                    self.queryLikelihoodModel.word_counts, self.queryLikelihoodModel.vocabs)

    def load_info(self):
        with open("data/translation.json", 'r') as outfile:
            self.translation_table = json.load(outfile)
        with open("data/word_count.json", 'r') as outfile:
            self.queryLikelihoodModel.word_counts = json.load(outfile)
        with open("data/vocab.json", 'r') as outfile:
            self.queryLikelihoodModel.vocabs = json.load(outfile)
        self.translation_model = TranslationModel(self.answer, self.λ, self.translation_table,
                                                self.queryLikelihoodModel.word_counts, self.queryLikelihoodModel.vocabs)

    def save_info(self):
        with open("data/translation.json", 'w+') as outfile:
            json.dump(self.translation_table, outfile)
        with open("data/word_count.json", 'w+') as outfile:
            json.dump(self.queryLikelihoodModel.word_counts, outfile)
        with open("data/vocab.json", 'w+') as outfile:
            json.dump(self.queryLikelihoodModel.vocabs, outfile)

    @staticmethod
    def build_query_likelihood_model(answers, mu: float) -> QueryLikelihoodModel:
        return QueryLikelihoodModel(answers, mu)

    def fetch_answer(self, question: str) -> str:
        pass

    def calculate_word_translation(self, threshold):
        pairs = list()
        for i in range(len(self.LMHranks)):
            for j in range(i, len(self.LMHranks[i])):
                if self.LMHranks[i][j] > threshold:
                    pairs.append((self.title[i].split(), self.title[j].split()))
        return WordTranslation.train(pairs)

    def find_relevant_questions(self, threshold: int) -> list:
        """
        Given a threshold value, find pairs LMHRank above threshold
        :param threshold:
        :type threshold:
        :return: list of pairs indices
        :rtype: list
        """
        pairs = list()
        for i in range(len(self.LMHranks)):
            for j in range(i, len(self.LMHranks[i])):
                if self.LMHranks[i][j] > threshold:
                    pairs.append((i, j))
        return pairs
