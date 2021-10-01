#functions

import os
import logging
import json
import numpy as np
from numpy.core.fromnumeric import argmax

#Enviorement 
env_row = 4
env_column = 5

episode = 10

# define alfa, gama,epsilon
alpha = 0.1
epsilon =  0.45
gamma =  0.9 
		
# define actions
actions = ['up', 'right', 'down', 'left']

# 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a).
# state -> env_row and env_column
states = env_row * env_column
q_table = np.zeros((states, 4))

print(q_table)

#location of the cheese
cheese_loc_row = env_row-1
cheese_loc_col = env_column-1

# rewards, 100 for cheese, -1 for each state
rewards = np.full((env_row,env_column),-1.)
rewards[cheese_loc_row, cheese_loc_col] = 100

def getAction(state):

    rand_num =  np.random.random()
    if rand_num < epsilon:
        print("act greedy")
        act = np.random.randint(4)
    else:    
        print("here is q table")
        print(q_table[state])
        act = np.argmax(q_table[state])

    return act

def takeAction(act,curr_row, curr_col):
    new_curr_row = curr_row
    new_curr_col = curr_col
    if act == 'up' and curr_row > 0:
        new_curr_row += 1
    elif act == 'right'  and curr_col < env_column-1:
        new_curr_col += 1 
    elif act == 'down'  and curr_row < env_row-1:
        new_curr_row -= 1 
    elif act == 'left'  and curr_col > 0:
        new_curr_col -= 1 
    return new_curr_row, new_curr_col

def isTerminate(stepList,lastNumbers):
    size = len(stepList)
    varValue = np.var(stepList[size-lastNumbers:])
    if varValue >=0 and varValue<0.1:
        return False
    return True	


def shortestPath(curr_row, curr_col):
    
    terminate = True
    episode = 0
    paths = []
    #for done in range(episode):
    while terminate:
        episode += 1
        curr_state = curr_row * curr_col
        steps = 0
        stepList = []
        pathtoTake = []
        pathtoTake.append([curr_row,curr_col])
        while rewards[curr_row,curr_col] != 100:
            act = getAction(curr_state)
            print(act)

            #update state
            old_curr_row, old_curr_col = curr_row, curr_col
            curr_row, curr_col = takeAction(act, curr_row, curr_col)
            new_state = curr_row * curr_col

            q_table[curr_state,act] = q_table[curr_state,act] + alpha*(rewards[curr_row,curr_col] + gamma*(np.max(q_table[new_state]) - q_table[curr_state]))
            
            epsilon -= epsilon*(0.000001)
            if epsilon < 0.01:
                epsilon = 0.01
            
            pathtoTake.append([curr_row,curr_col])
            steps += 1
        
        stepList.append(steps)
        paths.append(pathtoTake)
        terminate = isTerminate(stepList,20)


		
