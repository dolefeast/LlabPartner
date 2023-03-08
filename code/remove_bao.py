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
    print(omega)
    Om, OL, Ok = omega
    tag = f'_linspace_Om0{int(100*Om)}_OL0{int(100*OL)}.txt'
    k, pk = np.array(data[0]), np.array(data[1])
    k_lin = np.linspace(k[0], k[-1], len(k)) 

    print(f'\tPrinting to pklin_{tag}...')
    pk_lin = np.interp(k_lin, k, pk)
    pklin_df = pd.DataFrame([k_lin.T, pk_lin.T]).T
    pklin_df.to_csv(f'/home/santi/TFG/outputs_santi/linspace_class/pklin_{tag}', sep='\t', index=False, header=None)

    print(f'\tPrinting to psmlin_{tag}...')
    pk_nobao = util_tools.remove_bao(k, pk)
    psm_lin = np.interp(k_lin, k, pk_nobao)
    psm_df = pd.DataFrame([k_lin.T, psm_lin.T]).T
    psm_df.to_csv(f'/home/santi/TFG/outputs_santi/linspace_class/psmlin_{tag}', sep='\t', index=False, header=None)

    print(f'\tPrinting to Olin_{tag}...')
    olin = util_tools.calculate_olin(k, pk)
    olin = np.interp(k_lin, k, olin)
    olin_df = pd.DataFrame([k_lin.T, olin.T]).T
    olin_df.to_csv(f'/home/santi/TFG/outputs_santi/linspace_class/Olin_{tag}', sep='\t', index=False, header=None)

print('Done!')
