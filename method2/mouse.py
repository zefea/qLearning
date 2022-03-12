# 30.09.2021 
# genius_mouse python version 

from os import write

from matplotlib.pyplot import grid
from numpy import False_
#from src import *
from Train import *
from transfer import *

def writeOutput(filename,paths,obj):
    with open(filename, 'w') as f:
        f.write('Results\n')
        f.write(obj.toString())
        for x in paths:
            logging.info(x)
            f.write((str(x)))
            f.write(" --> number of steps: ")
            f.write(str(len(x)-1))
            f.write('\n')

def main():

    print("in main...")
    
    mainPath = os.getcwd() 
    path = mainPath + '/outputs/'
    fileName = path + "LogFile.log"

    logging.basicConfig(handlers=[logging.FileHandler(fileName, 'w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started')
    
    #Enviorement 
    env_row = 4
    env_column = 4
    #location of the cheese
    cheese_loc_row = env_row-1
    cheese_loc_col = env_column-1

    # rewards, 100 for cheese, -1 for each state
    rewards = np.full((env_row,env_column),-1)
    rewards[cheese_loc_row, cheese_loc_col] = 100

    banana = Train(rewards,env_row,env_column,decision='random',id=1,transfer=False)
    a = banana.training(0,0)  
    writeOutput(path + 'results.txt',a,banana)
    print("epsion of banana: ", banana.epsilon)

    '''
    cherry = Train(rewards,env_row,env_column,decision='softmax',id=2)
    d = cherry.training(0,0)
    writeOutput(path + 'results-2.txt',d,cherry)      
    print("epsion of banana: ", cherry.epsilon)

    #Plot Results
    gridInfo = str(env_row) + "x" + str(env_column)
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    banana.plotTraining()
    plt.subplot(1, 2, 2)
    cherry.plotTraining()
    plt.tight_layout()
    figureName = "grid" + gridInfo + '.png'
    plt.savefig(path + figureName)
    '''

    print("***********transfer****************")
    print(banana.getQtable())
    t = Transfer(banana.getQtable(),env_row,env_column)
    t.reverseTable()
    #print("555555555555555555")
    print(t.rev_table)

    #location of the cheese
    cheese_loc_row = env_row-1
    cheese_loc_col = 0

    # rewards, 100 for cheese, -1 for each state
    rewards = np.full((env_row,env_column),-1)
    rewards[cheese_loc_row, cheese_loc_col] = 100

    donkey = Train(rewards,env_row,env_column,decision='random',id=2,transfer=True)
    donkey.setTransferTable(t.rev_table) 
    #donkey.setEpsilon(0)
    
    print("donkey")
    print(donkey.transferTable)

    a2 = donkey.training(0,env_column-1) 

    print("Decision List")
    donkey.getDecisionList()
    print("epsion of donkey: ", donkey.epsilon)
    
    writeOutput(path + 'results' + str(donkey.id)+ '.txt',a2,donkey)



if __name__ == "__main__":
    main()