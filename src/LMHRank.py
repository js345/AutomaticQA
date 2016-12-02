'''
AutomaticQA LMHRank
Created on 11/30/16
@author: xiaofo
'''

from src.QueryLikelihoodModel import QueryLikelihoodModel


class LMHRANK:
    ranks = list()

    def __init__(self):
        pass

    @staticmethod
    def compute_scores(model: QueryLikelihoodModel) -> list:
        for doc in model.docs:
            pass