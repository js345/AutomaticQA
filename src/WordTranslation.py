'''
AutomaticQA WordTranslation
Created on 12/3/16
@author: xiaofo
'''

from nltk import IBMModel1
from nltk import AlignedSent
import json


class WordTranslation:

    @staticmethod
    def train(pairs):
        corpus = list()
        for q1, q2 in pairs:
            corpus.append(AlignedSent(q1, q2))
        em_ibm1 = IBMModel1(corpus, 20)
        with open("data/translation.json", 'w+') as outfile:
            json.dump(em_ibm1.translation_table, outfile)
        return em_ibm1.translation_table
