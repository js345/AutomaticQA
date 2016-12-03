'''
AutomaticQA LMHRank
Created on 11/30/16
@author: xiaofo
'''

from __future__ import division
from src.QueryLikelihoodModel import QueryLikelihoodModel


class LMHRANK:

    @staticmethod
    def compute_scores(model: QueryLikelihoodModel) -> list:
        """
        Returns LMHRanks on the upper right triangle of matrix
        :param model:
        :type model:
        :return:
        :rtype:
        """
        ranks = list()
        for index, doc in enumerate(model.docs):
            rank = model.retrieve_answers(doc)
            ranks.append(rank)
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks[i])):
                ranks[i][j] = (1 / ranks[i][j] + 1 / ranks[j][i]) / 2
        return ranks
