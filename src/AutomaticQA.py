'''
AutomaticQA AutomaticQA
Created on 11/30/16
@author: xiaofo
'''

from __future__ import division
from src.QueryLikelihoodModel import QueryLikelihoodModel
from src.LMHRank import LMHRANK
from src.WordTranslation import WordTranslation


class AutomaticQA:

    def __init__(self, data, mu):
        self.data = data
        self.title = data['Title']
        self.body = data['Body']
        self.answer = data['AnswerBody']
        self.queryLikelihoodModel = self.build_query_likelihood_model(self.answer, mu)
        self.LMHranks = LMHRANK.compute_scores(self.queryLikelihoodModel)

    def train(self):
        pass

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
        WordTranslation.train(pairs)

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
