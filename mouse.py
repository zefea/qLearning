# 30.09.2021 
# genius_mouse python version 

import os
import logging
import json
import numpy as np

#Enviorement 
env_row = 4
enc_column = 5

def main():
    # Initialize Log File Format
    path = os.getcwd()
    fileName = path + "/LogFile.log"
    print(fileName)
    logging.basicConfig(handlers=[logging.FileHandler(fileName, 'w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Started')
    


if __name__ == "__main__":
    main()