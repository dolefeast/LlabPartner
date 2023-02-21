import util_tools
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path


#TODO:
    #Given a directory full of Psmooth, return Olins and P_nobao.

#files = Path('/home/santi/TFG/outputs_santi/class_output/').glob('*pk.dat')
files = Path('/home/santi/TFG/outputs_santi/class_output').glob('*pk.dat')
files = util_tools.class_output
df, params = util_tools.many_files(files)      #all pk smooth as outputs from class

for data, omega in zip(df, params):
    Om, OL = omega
    k_in, pk_in = np.array(data[0]), np.array(data[1])
    pk_nobao = util_tools.remove_bao(k_in, pk_in)
    olin = util_tools.calculate_olin(k_in, pk_in)
    olin_df = pd.DataFrame([k_in, pk_in])
    olin_df.to_csv(f'/home/santi/TFG/class_output/Olin_Om{Om}_OL{OL}.txt', sep='\t', index=False)


plt.show()


