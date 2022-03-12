import codecs
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax


class Train:

    def __init__(self,rewards,env_row,env_column,decision,id,transfer):
      
        self.id = id
        self.rewards = rewards
        self.env_row = env_row
        self.env_column = env_column
        self.initQtable(env_row,env_column)
        self.decision = decision
        self.episode = 0 
        self.stepList = []
        self.transfer = transfer

        if transfer: 
            states = env_row * env_column
            self.transferTable = np.zeros((states, 4))

        #parameters
        self.var_number = 20
        self.epsilon =  0.2
        self.alpha = 0.1
        self.gamma =  0.9 
        self.probOfTransfer = 0.5
        # define actions
        self.actions = ['up', 'right', 'down', 'left']
		
    def getQtable(self):
        return self.q_table

    def setQtable(self,table):
        self.q_table = table

    def setEpsilon(self,eps):
        self.epsilon=eps

    def setTransferTable(self,table):
        self.transferTable = table

    def initQtable(self,row,column):
        states = row * column
        self.q_table = np.zeros((states, 4))

    def doTheMath(self,q_values,t):
        qALL = 0
        i=0
        dist = np.zeros((len(q_values), 2))

        for x in q_values:
            qALL += np.exp(x[1]/t)
        
        for y in q_values:
            dist[i,0] = np.exp(y[1]/t)/qALL
            dist[i,1] = y[0]
            i += 1 

        return np.flip(dist, axis=0)


    def numberOfMax(self,state,tempTable):
        
        max_val = max(tempTable[state])
        count = 0
        for i in range(4):
            val = tempTable[state,i]
            if val != max_val: 
                count += 1  

        return count, max_val

    def chooseWithSoftmax(self,state,tempTable): 
        
        diff_no, max_val = self.numberOfMax(state,tempTable)  
        if diff_no != 0:
            q_values = np.zeros((diff_no, 2))     #4 actions, each has index and q value of themselves
        else: 
            q_values = np.zeros((4, 2))

        idx = 0
        for i in range(4):
            #print(self.q_table[state,i])
            val = tempTable[state,i]
            if diff_no == 4:
                q_values[idx,0] = i
                q_values[idx,1] = val  
                idx += 1
            else:
                if val != max_val: 
                    q_values[idx,0] = i
                    q_values[idx,1] = val  
                    idx += 1

        sortedQ = np.core.records.fromarrays(q_values.transpose(), names="col1, col2")
        sortedQ.sort(order="col2")
        probs = self.doTheMath(sortedQ,5)

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

    def getAction(self,state):

        rand_num =  np.random.random()
        if rand_num < self.epsilon:
            if self.decision == 'softmax': 
                act = self.chooseWithSoftmax(state,self.q_table)
            else: 
                #print("act greedy")
                act = np.random.randint(4)
        else:    
            #print("here is q table")
            #print(q_table[state])
            act = np.argmax(self.q_table[state])

        return act

    def getActionWithTransfer(self,state):

        rand_num =  np.random.random()
        if rand_num < self.probOfTransfer:
            #print("you learned it before come on!!")
            #act = self.chooseWithSoftmax(state,self.transferTable)
            act = np.argmax(self.transferTable[state])
            #print("the act is :" + str(act))
        else:
            act = self.getAction(state)

        return act


    def takeAction(self,act,curr_row, curr_col):
        new_curr_row = curr_row
        new_curr_col = curr_col
        if self.actions[act] == 'up' and curr_row > 0:
            new_curr_row -= 1
        elif self.actions[act] == 'right'  and curr_col < self.env_column-1:
            new_curr_col += 1 
        elif self.actions[act] == 'down'  and curr_row < self.env_row-1:
            new_curr_row += 1 
        elif self.actions[act] == 'left'  and curr_col > 0:
            new_curr_col -= 1 

        return new_curr_row, new_curr_col

    def isTerminate(self,stepList,lastNumbers):
        size = len(stepList)
        logging.info(str(stepList[size-lastNumbers:]))
        varValue = np.var(stepList[size-lastNumbers:])
        logging.info("variance: " + str(varValue))
        if varValue >=0 and varValue<0.1:
            return False
        return True	

    def decreaseEpsilon(self):
        x = self.epsilon
        if x < 0.01:
            return 0.01
        else:
            x -= x*(0.000001)

        self.epsilon = x

    

    def training(self,startx,starty):
    
        print("Training begins")
        terminate = True
        episode = 0
        paths = []
        stepList = []

        #for done in range(episode_number):
        while terminate:
            episode += 1
            c = "----------episode " + str(episode) + "-----------"
            logging.info(c)
            curr_row, curr_col = startx,starty
            curr_state = curr_row * self.env_column + curr_col 
            steps = 0
            pathtoTake = []
            pathtoTake.append([curr_row,curr_col])
            
            while self.rewards[curr_row,curr_col] != 100:
                
                if self.transfer == True:
                    act = self.getActionWithTransfer(curr_state)
                else:
                    act = self.getAction(curr_state)

                a = "take " + str(self.actions[act])
                #update state
                old_curr_row, old_curr_col = curr_row, curr_col
                curr_row, curr_col = self.takeAction(act, curr_row, curr_col)
                output = "(" + str(old_curr_row) + "," +  str(old_curr_col) + ")" + "--" + a + "--" + "(" + str(curr_row) + "," + str(curr_col) + ")"
                #logging.info(output)
                new_state = curr_row *self.env_column + curr_col

                #q_table[curr_state,act] = q_table[curr_state,act] + alpha * (rewards[curr_row,curr_col] + gamma * (np.max(q_table[new_state]) - q_table[curr_state]))
                
                #receive the reward for moving to the new state, and calculate the temporal difference
                reward = self.rewards[curr_row,curr_col]
                old_q_value = self.q_table[curr_state, act]
                temporal_difference = reward + (self.gamma * np.max(self.q_table[new_state])) - old_q_value
                
                #update the Q-value for the previous state and action pair
                new_q_value = old_q_value + (self.alpha * temporal_difference)
                self.q_table[curr_state, act] = new_q_value
                curr_state = new_state 
                #print(q_table[curr_state, act])
                self.decreaseEpsilon()
                
                pathtoTake.append([curr_row,curr_col])
                steps += 1
            
            stepList.append(steps)
            b = "Number of steps: " + str(steps)
            logging.info(b)
            #logging.info(str(pathtoTake))
            #logging.info(str(epsilon))
            paths.append(pathtoTake)
            #print(len(stepList))
            if len(stepList) > self.var_number:
                terminate = self.isTerminate(stepList,self.var_number)
                #logging.info(q_table)
            
        size = len(paths)
        self.episode = episode
        self.stepList = stepList

        return paths[size-5:]

    def plotTraining(self):
         # plotting the points
        x = range(self.episode)
        plt.plot(x,self.stepList)
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Step number')
        plt.title(self.decision + ' selection - ' + str(self.id))
        mainPath = os.getcwd() 
        path = mainPath + '/outputs/'
        
    def toString(self):
        gridInfo = "Grid: " + str(self.env_row) + "x" + str(self.env_column)
        parameterInfo = "Explorative selection: " + self.decision
        episodeInfo = "Number of episodes: " + str(self.episode)

        return gridInfo + "\n" +parameterInfo + "\n" + episodeInfo + "\n"
