#functions

import os
import logging
import codecs, json 
import numpy as np
from numpy.core.fromnumeric import argmax

#Enviorement 
env_row = 4
env_column = 5

episode_number = 1
var_number = 20
# define alfa, gama,epsilon
alpha = 0.1
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
rewards = np.full((env_row,env_column),0)
rewards[cheese_loc_row, cheese_loc_col] = 100

print(type(rewards[0,0]))

def getAction(state,epsilon):

    rand_num =  np.random.random()
    if rand_num < epsilon:
        #print("act greedy")
        act = np.random.randint(4)
    else:    
        #print("here is q table")
        #print(q_table[state])
        act = np.argmax(q_table[state])

    return act

def takeAction(act,curr_row, curr_col):
    new_curr_row = curr_row
    new_curr_col = curr_col
    if actions[act] == 'up' and curr_row > 0:
        new_curr_row -= 1
    elif actions[act] == 'right'  and curr_col < env_column-1:
        new_curr_col += 1 
    elif actions[act] == 'down'  and curr_row < env_row-1:
        new_curr_row += 1 
    elif actions[act] == 'left'  and curr_col > 0:
        new_curr_col -= 1 

    return new_curr_row, new_curr_col

def isTerminate(stepList,lastNumbers):
    size = len(stepList)
    logging.info(str(stepList[size-lastNumbers:]))
    varValue = np.var(stepList[size-lastNumbers:])
    logging.info("variance: " + str(varValue))
    if varValue >=0 and varValue<0.1:
        return False
    return True	

def decreaseEpsilon(x):
    
    if x < 0.01:
        return 0.01
    else:
        x -= x*(0.000001)
    return x


def training(curr_row, curr_col):
    
    print("Training begins")
    terminate = True
    episode = 0
    paths = []
    stepList = []
    epsilon =  0.2

    #for done in range(episode_number):
    while terminate:
        episode += 1
        c = "----------episode " + str(episode) + "-----------"
        logging.info(c)
        curr_row, curr_col = 0,0
        curr_state = curr_row *env_column + curr_col 
        steps = 0
        pathtoTake = []
        pathtoTake.append([curr_row,curr_col])
        
        while rewards[curr_row,curr_col] != 100:
            
            act = getAction(curr_state,epsilon)
            a = "take " + str(actions[act])
            #update state
            old_curr_row, old_curr_col = curr_row, curr_col
            curr_row, curr_col = takeAction(act, curr_row, curr_col)
            output = "(" + str(old_curr_row) + "," +  str(old_curr_col) + ")" + "--" + a + "--" + "(" + str(curr_row) + "," + str(curr_col) + ")"
            #logging.info(output)
            new_state = curr_row *env_column + curr_col

            #q_table[curr_state,act] = q_table[curr_state,act] + alpha * (rewards[curr_row,curr_col] + gamma * (np.max(q_table[new_state]) - q_table[curr_state]))
            
            #receive the reward for moving to the new state, and calculate the temporal difference
            reward = rewards[curr_row,curr_col]
            old_q_value = q_table[curr_state, act]
            temporal_difference = reward + (gamma * np.max(q_table[new_state])) - old_q_value
            
            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (alpha * temporal_difference)
            q_table[curr_state, act] = new_q_value
            curr_state = new_state 
            #print(q_table[curr_state, act])
            epsilon = decreaseEpsilon(epsilon)
            
            pathtoTake.append([curr_row,curr_col])
            steps += 1
        
        stepList.append(steps)
        b = "Number of steps: " + str(steps)
        logging.info(b)
        #logging.info(str(pathtoTake))
        #logging.info(str(epsilon))
        paths.append(pathtoTake)
        #print(len(stepList))
        if len(stepList) > var_number:
            terminate = isTerminate(stepList,var_number)
            #logging.info(q_table)
        
    size = len(paths)
    
    return q_table, paths[size-5:]



		
