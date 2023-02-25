import util_tools
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path

#TODO:
    #Given a directory full of Psmooth, return Olins and P_nobao.

#files = Path('/home/santi/TFG/outputs_santi/class_output/').glob('*pk.dat')
files = list(Path('/home/santi/TFG/outputs_santi/class_outputs').glob('*pk*'))
hector_files = list(Path('/home/santi/TFG/lrg_eboss/model/').glob('Pk*'))
print('Opening Hector Pk')
hector_df, _ = util_tools.many_files(hector_files)      #all pk smooth as outputs from class
print('Which of your files do you want to open?')
df, params = util_tools.many_files(files)
h = 0.676

for data, param in zip(df, params):
    k_in, pk_in = data[0], data[1]
    y = pk_in#util_tools.remove_bao(k_in, pk_in)
    plt.plot(data[0],y, label=param)
    


k, pk_in = hector_df[0][0], hector_df[0][1]
plt.plot(k, pk_in, '--k')
plt.show()

if len(df)==1:
    x1, y1 = k, pk_in
    x2, y2 = data[0], data[1]
    ymod=np.interp(x2, x1, y1)
    #ymod = ymod(x2)
    plt.plot(x2, (y2-ymod)/y2, label='Comparison')
    plt.legend(loc='best')
    plt.show()
