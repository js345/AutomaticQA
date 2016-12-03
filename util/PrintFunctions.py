'''
AutomaticQA PrintFunctions
Created on 12/2/16
@author: xiaofo
'''


def show_relevant_questions(pairs, automaticQA):
    for i, j in pairs:
        print("Q1")
        print(automaticQA.title[i])
        print("Q2")
        print(automaticQA.title[j])
        print("--------------------")