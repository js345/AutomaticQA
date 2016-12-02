'''
AutomaticQA LMHRank
Created on 11/30/16
@author: xiaofo
'''


class LMHRANK:
    ranks = list()

    def __init__(self):
        pass

    @staticmethod
    def compute_scores(model):
        for doc in model.docs:
            pass