'''
AutomaticQA AutomaticQA
Created on 11/30/16
@author: xiaofo
'''

from src.QueryLikelihoodModel import QueryLikelihoodModel


class AutomaticQA:

    def __init__(self, data, mu):
        self.data = data
        self.title = data['Title']
        self.body = data['Body']
        self.answer = data['AnswerBody']
        self.queryLikelihoodModel = self.build_rank(self.answer, mu)
        print(self.queryLikelihoodModel.retrieve_answers(self.answer[0]))

    def train(self):
        pass

    @staticmethod
    def build_rank(answers, mu: float) -> QueryLikelihoodModel:
        return QueryLikelihoodModel(answers, mu)

    def fetch_answer(self, question: str) -> str:
        pass

    def calculate_answer_similarity(self):
        pass
