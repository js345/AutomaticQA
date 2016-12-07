'''
AutomaticQA WordTranslation
Created on 12/3/16
@author: xiaofo
'''

from nltk import IBMModel1
from nltk import AlignedSent


class WordTranslation:

    @staticmethod
    def train(pairs):
        corpus = list()
        for q1, q2 in pairs:
            corpus.append(AlignedSent(q1, q2))
        em_ibm1 = IBMModel1(corpus, 20)
        print(em_ibm1.translation_table)
        return em_ibm1.translation_table
