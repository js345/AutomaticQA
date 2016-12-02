'''
AutomaticQA QueryLikelihoodModel
Created on 11/30/16
@author: xiaofo
'''

from __future__ import division
import numpy as np
import scipy.stats as ss


class QueryLikelihoodModel:
    '''
    Query Likelihood Model with Dirichlet Smoothing
    '''

    def __init__(self, docs, mu):
        self.docs = docs
        self.word_num = sum(len(doc.split()) for doc in docs)
        self.mu = mu
        self.vocabs = dict()
        self.word_counts = list()
        self.build_model()

    def build_model(self):
        '''
        Build the model by counting word occurrences
        :return:
        :rtype:
        '''
        for doc in self.docs:
            word_count = dict()
            for word in doc.split():
                self.vocabs[word] = self.vocabs.get(word, 0) + 1
                word_count[word] = word.get(word, 0) + 1
            self.word_counts.append(word_count)

    def retrieve_answers(self, question):
        question = question.split()
        query_count = dict()
        for word in question:
            query_count[word] = query_count.get(word, 0) + 1
        scores = list()
        for i, doc in enumerate(self.docs):
            score = 0
            for word in query_count:
                smoothed_value = self.word_counts[i].get(word, 0) + self.mu * self.vocabs[word] / self.word_num
                smoothed_value /= len(self.docs[i].split()) + self.mu
                score += query_count[word] * np.log(smoothed_value)
            scores.append(score)
        return ss.rankdata(scores)
