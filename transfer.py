import codecs
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax
import math

class Transfer:
    def __init__(self,qtable,row,column):
         self.temp_table = qtable 
         self.initRevtable(row,column)
         self.statePairs = []
         self.row = row
         self.column = column

    def setRevTable(self,table):
        self.rev_table = table

    def initRevtable(self,row,column):
        states = row * column
        self.rev_table = np.zeros((states, 4))


    # which state is symatric with what state?
    def defineRelation(self,row, column):
        
        pairList = []
        for i in range(int(column/2)): 
            print("******************") 
            curr_col=i #current column no
            
            for j in range(row):
                reversedCurr_col=column
                pairs = []
                pairs.append(curr_col)
                reversedCurr_col = reversedCurr_col * (j+1)
                pairs.append(reversedCurr_col-i-1)
                print(pairs)
                pairList.append(pairs)
                curr_col = curr_col + column
        
        self.statePairs = pairList
      
    def reverseTable(self):

        self.defineRelation(self.row, self.column)
        states = self.row * self.column
        q_table = np.zeros((states, 4))

        table = []
        lenOfStates = len(self.statePairs)
        for n in range(lenOfStates):
            state1 = self.statePairs[n][0]
            state2 = self.statePairs[n][1]

            #swap(state1,state2)
            q_table[state1] =  self.temp_table[state2]
            temp = q_table[state1,1]
            q_table[state1,1] = q_table[state1,3]
            q_table[state1,3] = temp
            
            q_table[state2] =  self.temp_table[state1]
            temp = q_table[state2,1]
            q_table[state2,1] = q_table[state2,3]
            q_table[state2,3] = temp

        if (self.column % 2) == 1:
            for i in range(self.row):
                state = i * self.column + math.floor(self.column/2)
                print("state:::")
                print(state)
                q_table[state] = self.temp_table[state]

        self.setRevTable(q_table)








      