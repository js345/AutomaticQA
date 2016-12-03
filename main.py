'''
AutomaticQA main
Created on 11/25/16
@author: xiaofo
'''

from src.AutomaticQA import AutomaticQA
from util.Dataloader import load
from util.PrintFunctions import show_relevant_questions

path = 'data/QueryResults.csv'
data = load(path)

automaticQA = AutomaticQA(data, 50)
pairs = automaticQA.find_relevant_questions(0.05)

show_relevant_questions(pairs)

