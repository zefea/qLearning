# 30.09.2021 
# genius_mouse python version 

from src import *


def main():

    print("in main...")
    for row in rewards:
        print(row)

    path = os.getcwd()
    fileName = path + "/LogFile.log"
    print(fileName)
    logging.basicConfig(handlers=[logging.FileHandler(fileName, 'w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started')
    a,b = training(0,0)
    logging.info(a)

    for i in range(len(b)):
        logging.info("Path: " + str(b[i]))

   

if __name__ == "__main__":
    main()