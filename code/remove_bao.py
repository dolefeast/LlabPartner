import util_tools
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path

#TODO:
    #Given a directory full of Psmooth, return Olins and P_nobao.

#files = Path('/home/santi/TFG/outputs_santi/class_output/').glob('*pk.dat')
files = list(Path('/home/santi/TFG/outputs_santi/class_outputs').glob('*pk.dat'))
#hector_files = list(Path('/home/santi/TFG/lrg_eboss/model/Olinkirkby_eboss_comb_z070_matterpower_15.txt').glob('Olin*'))
df, params = util_tools.many_files(files)      #all pk smooth as outputs from class

print('Creating Olin data files...')
for data, omega in zip(df, params):
    Om, OL, Ok = (0.31,0.69,0.)
    k_in, pk_in = np.array(data[0]), np.array(data[1])
    pk_nobao = util_tools.remove_bao(k_in, pk_in)
    olin = util_tools.calculate_olin(k_in, pk_in)
    olin_df = pd.DataFrame([k_in.T, olin.T]).T
    plt.plot(k_in, olin_df[0], '+')
    print(f'\tPrinting to Olin_Om0{int(100*Om)}_OL0{int(100*OL)}.txt...')
    olin_df.to_csv(f'/home/santi/TFG/outputs_santi/class_outputs/Olin_np_Om0{int(100*Om)}_OL0{int(100*OL)}.txt', sep='\t', index=False)

print('Done!')
plt.show()
