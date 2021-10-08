# 30.09.2021 
# genius_mouse python version 

from os import write
from src import *



def writeOutput(filename,b):
    with open(filename, 'w') as f:
        f.write('Results\n')
        for x in b:
            logging.info(x)
            f.write((str(x)))
            f.write(" --> number of steps: ")
            f.write(str(len(x)-1))
            f.write('\n')

def main():

    print("in main...")
    for row in rewards:
        print(row)

    mainPath = os.getcwd() 
    path = mainPath + '/outputs/'
    fileName = path + "LogFile.log"

    logging.basicConfig(handlers=[logging.FileHandler(fileName, 'w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started')

    #training --> q-table and last 5 path
    a,b,c = training(0,0,decision='softmax3')
    logging.info(a)
    
    for i in range(10):
        testing(0,0,c)

    writeOutput(path + 'results.txt',b)
            
    
if __name__ == "__main__":
    main()