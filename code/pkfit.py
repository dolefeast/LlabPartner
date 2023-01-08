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

df = util_tools.select_files(util_tools.files)  #List files
df, names = util_tools.open_file(df)            #Open file and return names

k = df.loc[:, 0].values
mPk = df.loc[:, 1].values

kmin, kmax = (2.8e-2, 4.5e-1)

index = np.where(np.logical_and(k>=kmin, k<=kmax))
k_small = k[index] #For plotting purposes
mPk_small = mPk[index]

remove_bao = util_tools.remove_bao(k, mPk)
remove_bao_small = remove_bao[index]
only_bao = mPk_small - remove_bao_small
X, Y = util_tools.peaks_fit(k_small, only_bao)

fft_only_bao = sp.fft.fft(only_bao)
bao_fft = util_tools.fft_bao(k, mPk)    #To check the difference between
                                        #taking the fft of the whole picture
                                        #vs the interesting section
bao_fft_small = util_tools.fft_bao(k_small, mPk_small)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12.,9.], dpi=100)
#ax1.set_xlabel(names[0])
#ax1.set_ylabel(names[1])
ax1.set_title('Only BAOs')
ax2.set_xlabel('Mpc/h')
ax2.set_title(r'$\int_0^\infty dk k^2 \frac{\sin(kr)}{kr}$')
#ax2.set_xscale('log')
#ax2.set_yscale('log')

[ax1.plot(x, y, '--k', linewidth=0.855555, alpha=0.9) for x, y in zip(X, Y)]
ax1.plot(k_small, only_bao, label='Only BAOs')
ax1.legend(loc='best')
ax2.plot(*bao_fft, label='Fourier transform')
ax2.plot(*bao_fft_small, label='Fourier transform small')
ax2.legend(loc='best')

fig.savefig('./fig/mpk.pdf')
plt.show()
