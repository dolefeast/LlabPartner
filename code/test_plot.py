import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import util_tools
from scipy.signal import find_peaks, argrelmin
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False #Latex format stuff
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

df, _ = util_tools.many_files(util_tools.model)  #List files
#df, names = util_tools.open_file(df)            #Open file and return names


fig, ax1 = plt.subplots(1, 1, figsize=[16.,9.], dpi=100)
#ax1.set_xlabel(names[0])
#ax1.set_ylabel(names[1])
#ax2.set_xlabel('Mpc/h')
#ax2.set_title(r'$\int_0^\infty dk k^2 P(k) \frac{\sin(kr)}{kr}$')
ax1.set_xscale('log')
ax1.set_yscale('log')

#ax1.plot(k_small, only_bao, color='steelblue', alpha=0.9) ax1.plot(datA[0], data[1], '--k', linewidth=0.7, alpha=0.7)
for data in df:
    ax1.plot(data[0], data[1], '--', linewidth=0.7, alpha=0.7)

ax1.legend(loc='best')
#ax1.plot(X, Y, '--k', linewidth=0.8, alpha=0.9)
#ax1.legend(loc='best')
#ax1.plot(R, Xi)
#ax2.set_xlim((0, 40))
#ax2.legend(loc='best')

fig.savefig('./fig/mpk.pdf')
plt.show()
