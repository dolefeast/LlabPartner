import util_tools
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth=1.7)
import matplotlib
#matplotlib.use('pgf') #Saves the output as pgf
matplotlib.rcParams['axes.unicode_minus'] = False #Latex format stuff
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


#files = Path('/home/santi/TFG/outputs_santi/class_output/').glob('*pk.dat')
files = list(Path('/home/santi/TFG/outputs_santi/linspace_class').glob('*'))
hector_files = list(Path('/home/santi/TFG/lrg_eboss/model/').glob('*'))
print('Opening Hector Pk')
hector_df, _ = util_tools.many_files(hector_files)      #all pk smooth as outputs from class
print('Which of your files do you want to open?')
df, params = util_tools.many_files(files)
h = 0.676

fig, ax =plt.subplots(1, 2)
fig.tight_layout()
for data, param in zip(df, params):
    k_in, pk_in = data[0], data[1]
    y = pk_in#util_tools.remove_bao(k_in, pk_in)
    ax[0].plot(data[0],y, color='teal')

k_in, pk_in = hector_df[0][0], hector_df[0][1]
ax[0].plot(k_in, pk_in, '--k')
plt.legend(loc='best')

if len(df)==1:
    x1, y1 = k_in, pk_in
    x2, y2 = data[0], data[1]
    ymod=np.interp(x2, x1, y1)
    #ymod = ymod(x2)
    ax[1].plot(x2, (y2-ymod)/ymod*100, color='teal')
    ax[1].set_ylabel(r'$100\cdot\frac{P_{eBoss}(k) - P_{Santi}(k)}{P_{eBoss}(k)}\%$')
    ax[1].set_xlabel(r'k [h/Mpc]')
#    plt.plot(x2, (1-0.665)*y2, label='My data')
#    plt.plot(x1, y1, label='Original Y')
    #plt.savefig('../figs/Olin_relative_comparison.pdf')
    plt.show()
