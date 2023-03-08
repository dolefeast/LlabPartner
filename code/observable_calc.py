import matplotlib as mpl
from pathlib import Path
import util_tools
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as ct
import scipy as sp

zmax = 0.698
H0 = 67.6
rs = 147.784

def grouped_weighted_avg(values, weights, by):
    return (values * weights).groupby(by).sum() / weights.groupby(by).sum()

files = Path('~/TFG/outputs_santi/class_Om031_OL069/logfiles/').glob('*')
#calculate_observables(*params[0])
Ok_list = []
alpha_para_list = []
alpha_perp_list = []

for frame, omegas in zip(df, params):
    
    Ok = omegas[-1]
    alpha_para, alpha_para_std = util_tools.weighted_avg_and_std(frame[2], frame[0])
    alpha_perp, alpha_perp_std = util_tools.weighted_avg_and_std(frame[3], frame[0])
    if Ok in Ok_list:
        continue

    Ok_list.append(Ok)
    alpha_para_list.append((alpha_para, alpha_para_std))
    alpha_perp_list.append((alpha_para, alpha_para_std))

  #  print(f'Ok: {Ok}, alpha_para_wt:{alpha_para_mean_wt}\n\t alpha: {alpha_para_mean}\n')

fig, ((ax1, ax2),(ax12, ax22),(ax13, ax23)) = plt.subplots(3, 2, sharex=True, figsize=(10, 7))
H = lambda z, Ok, Om=0.31: H0*np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + 1-Ok-Om)
DH_fid = lambda z, Ok: ct.c/1000/H(z, Ok)
Ok_cont = np.linspace(-0.15,0.15,100)
DA_fid = np.array([sp.integrate.quad(DH_fid, 0, zmax, args=(ok,))[0] for ok in Ok_cont])
for Ok, apara, aperp in zip(Ok_list, alpha_para_list, alpha_perp_list):
    ax1.errorbar(Ok, apara[0], yerr=apara[1], fmt='xb', elinewidth=0.7)
    ax2.errorbar(Ok, aperp[0], yerr=aperp[1], fmt='xb', elinewidth=0.7)
    ax13.errorbar(Ok, DH_fid(zmax, Ok)*apara[0]/rs,  yerr=DH_fid(zmax, Ok)*apara[0]/rs, fmt='xb', elinewidth=0.7) 
    ax23.errorbar(Ok, DA_fid[int(100*(Ok+0.15)/0.3)]*aperp[0]/rs,  yerr=DA_fid[int(100*(Ok+0.15)/0.3)]*aperp[0]/rs, fmt='xb', elinewidth=0.7) 


ax12.plot(Ok_cont, DH_fid(zmax, Ok_cont)/rs, linewidth=0.7)
ax22.plot(Ok_cont, DA_fid/rs, linewidth=0.7)
ax1.set_ylabel(r'$\alpha_{para}$'), ax2.set_ylabel(r'$\alpha_{perp}$')
ax12.set_ylabel(r'$\left[ DH/r_s\right]_{fid}$'), ax22.set_ylabel(r'$\left[ DA/r_s\right]_{fid}$')
ax13.set_ylabel(r'$DH/r_s$'), ax23.set_ylabel(r'$DA/r_s$')
ax13.set_xlabel(r'$\Omega_k$'), ax23.set_xlabel(r'$\Omega_k$')
plt.tight_layout()
plt.savefig('/home/santi/TFG/figs/DA_DH.pdf')
plt.show()


