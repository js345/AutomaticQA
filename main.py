'''
AutomaticQA main
Created on 11/25/16
@author: xiaofo
'''

from src.Dataloader import load
from src.AutomaticQA import AutomaticQA

path = 'data/QueryResults.csv'
data = load(path)

automaticQA = AutomaticQA(data, 50)
