'''
AutomaticQA TranslationModel
Created on 12/3/16
@author: xiaofo
'''


from __future__ import division
import numpy as np


class TranslationModel:

    def __init__(self, docs, λ, translation_table, word_counts, vocabs):
        self.docs = docs
        self.λ = λ
        self.translation_table = translation_table
        self.word_counts = word_counts
        self.vocabs = vocabs

    def retrieve_answers(self, query: str) -> list:
        """
        Given a query retrieve
        :param query:
        :type query:
        :return:
        :rtype:
        """
        query = query.split()
        query_count = dict()
        for word in query:
            query_count[word] = query_count.get(word, 0) + 1
        scores = list()
        for i, doc in enumerate(self.docs):
            score = 0
            for word in query_count:
                smoothed_value = (1 - self.λ) * sum(self.translation_table[term].get(word, 0) * self.word_counts[i] for term in self.translation_table)
                smoothed_value += self.λ * self.vocabs[word]
                score += query_count[word] * np.log(smoothed_value)
            scores.append(score)
        return scores

