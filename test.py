 # plotting the points
import os
import logging
import codecs, json 
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
    
x = [1,2,3,4,5,6,7,8,9,10]
y = [11,34,77,433,22,88,77,22,66,45]

plt.plot(x,y)
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Step number')

plt.title('Episode & Steps')

plt.show()