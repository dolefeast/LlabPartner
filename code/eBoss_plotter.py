import matplotlib.pyplot as plt
import numpy as np
import util_tools
from pathlib import Path
import pandas as pd

files = list(Path('/home/santi/TFG/DATA/rustico_output').glob('Power_*'))
#file_name = util_tools.select_files(files)
kmin, kmax = (0., 0.16)

print('Elige los datos santi')
df_list_santi, param_list_santi = util_tools.many_files(files)
print('Elige los datos hector')
df_list, param_list = util_tools.many_files(files)  
#This could be made automatic if I had 2 dirs one for my outputs and #one 
#for hector's and when selecting a certain file with certain parameters
#it searches for the same parameters in hector's dir. Do if bored

#idx = np.where(np.logical_and(df_list_santi[0][1]>=kmin, df_list_santi[0][1]<=kmax))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
#ax1.set_xlim((0,0.16)), ax2.set_xlim((0,0.16))

for data, params in zip(df_list_santi, param_list_santi):
    kk, P0, P2 = data[1], data[2], data[3] 
    ax1.plot(kk, kk*P0, label=str(params)+' santi')
    ax2.plot(kk, kk*P2, label=str(params)+' santi')

for data, params in zip(df_list, param_list):
    kk, P0, P2 = data[1], data[2], data[3] 
    ax1.plot(kk, kk*P0, '--', linewidth=0.8, label=str(params)+' hector')
    ax2.plot(kk, kk*P2, '--', linewidth=0.8, label=str(params)+' hector')

ax1.set_title('$k*P_0(k)$'), ax2.set_title('$k*P_2(k)$')
ax1.legend(loc='best'), ax2.legend(loc='best')
plt.savefig('/home/santi/TFG/figs/hectors_vs_mydata.png')
plt.show()

