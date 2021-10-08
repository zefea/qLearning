#functions

import codecs
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax

#Enviorement 
env_row = 5
env_column = 5
episode_number = 1
var_number = 20
#parameters
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
rewards = np.full((env_row,env_column),-1)
rewards[cheese_loc_row, cheese_loc_col] = 100

print(type(rewards[0,0]))

def doTheMath(q_values,t):
    qALL,i = 0
    dist = np.zeros((len(q_values), 2))

    for x in q_values:
        print("addition of value: ",x[1])
        qALL += np.exp(x[1]/t)
	 
    for y in q_values:
        print("the value:",y[1])
        dist[i,0] = np.exp(y[1]/t)/qALL
        dist[i,1] = y[0]
        i += 1 
      
    print("distrubition with softmax")
    print(dist)

    return np.flip(dist, axis=0)

def numberOfMax(state):
    max_val = max(q_table[state])
    print("max number is ", max_val)
    print(q_table[state]) 
    count = 0
    for i in range(4):
        val = q_table[state,i]
        if val != max_val: 
            count += 1  
    
    print("count : ", count)
    return count, max_val

def chooseWithSoftmax(state): 
    
    diff_no, max_val = numberOfMax(state)  
    q_values = np.zeros((diff_no, 2))     #4 actions, each has index and q value of themselves

    idx = 0
    for i in range(4):
        print(q_table[state,i])
        val = q_table[state,i]
        if val != max_val: 
            q_values[idx,0] = i
            q_values[idx,1] = val  
            idx += 1

    sortedQ = np.core.records.fromarrays(q_values.transpose(), names="col1, col2")
    sortedQ.sort(order="col2")
    print("SORTED")
    print(sortedQ)

    probs = doTheMath(sortedQ,5)
    print("Probs")
    print(probs)

    selectAction = -1
    rand_num =  np.random.random()
    probMax = probs[0,0]
    probMin = 0

    for i in range(len(probs)):
        if rand_num > probMin and rand_num < probMax:
            selectAction = probs[i,1]
            break
        else:
            probMin = probMax
            probMax = probs[i+1,0] + probMin

    return int(selectAction)

def getAction(state,epsilon,decision):

    rand_num =  np.random.random()
    if rand_num < epsilon:
        if decision == 'softmax': 
            act = chooseWithSoftmax(state)
        else: 
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

def training(curr_row, curr_col,decision):
    
        print("Training begins")
        terminate = True
        episode = 0
        paths = []
        stepList = []
        epsilon =  0.25

        #for done in range(episode_number):
        while terminate:
            episode += 1
            c = "----------episode " + str(episode) + "-----------"
            logging.info(c)
            curr_row, curr_col = 0,0
            curr_state = curr_row * env_column + curr_col 
            steps = 0
            pathtoTake = []
            pathtoTake.append([curr_row,curr_col])
            
            while rewards[curr_row,curr_col] != 100:
                
                act = getAction(curr_state,epsilon,decision)
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
        
        # plotting the points
        x = range(episode)
        plt.plot(x,stepList)
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Step number')
        plt.title('Episode & Steps')
        mainPath = os.getcwd() 
        path = mainPath + '/outputs/'
        
        plt.savefig(path + 'figure.png')
        plt.show()
        
        return q_table, paths[size-5:], epsilon

def testing(startx,starty,epsilon):
    print("hello test time")
    c = "----------Testing-----------"
    logging.info(c)
    curr_row, curr_col = startx,starty
    curr_state = curr_row *env_column + curr_col 
    steps = 0
    pathtoTake = []
    pathtoTake.append([curr_row,curr_col])
    while rewards[curr_row,curr_col] != 100:
        
        act = getAction(curr_state,epsilon,decision='random')
        a = "take " + str(actions[act])
        #update state
        old_curr_row, old_curr_col = curr_row, curr_col
        curr_row, curr_col = takeAction(act, curr_row, curr_col)
        new_state = curr_row *env_column + curr_col
        curr_state = new_state 

        pathtoTake.append([curr_row,curr_col])
        steps += 1
    
    b = "Number of steps: " + str(steps)
    logging.info(b)
    print(b)
  


		
