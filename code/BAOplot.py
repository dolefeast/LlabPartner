#!/usr/bin/env python3

#A script that runs an interactive plotter 
#in any dir.
#TODO

import util_tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys

#Normalize everything to its max value so the study 
#of the curves' behavious is easier
normalise = '-n' in sys.argv

filtr = input('Give a filter... (Nothing or "*" for no filter)')
if filtr == '':
    filtr = '*'

files = list(Path('.').glob('*'+filtr+'*'))
df, names = util_tools.many_files(files)

for data, paramname in zip(df, names):
    x, y = data[0], data[1]
    if normalise:
        y /= np.max(y)

    if paramname != 1:
        plt.plot(x, y, label=paramname)
    else:
        plt.plot(x, y)

plt.show()
