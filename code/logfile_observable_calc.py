import matplotlib as mpl
from pathlib import Path
import re
import util_tools
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as ct
import scipy as sp
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


zmax = 0.698
H0 = 67.6
rs = 147.784

def grouped_weighted_avg(values, weights, by):
    return (values * weights).groupby(by).sum() / weights.groupby(by).sum()

files = list(Path('/home/santi/TFG/outputs_santi/class_Om031_OL069/logfiles').glob('*'))
#calculate_observables(*params[0])
Ok_list = []
a_para = []
a_perp = []

for data in files:
    omegas = util_tools.get_params(str(data.stem))
    Ok = omegas[-1]
    if Ok in Ok_list:
        continue

    with data.open() as open_data:
        for i, line in enumerate(open_data):
            if i == 9:
                p = re.compile('[0-9].[0-9]*e-[0-9]*')
                matches = p.findall(line)
                a_para.append([float(x) for x in matches])
            elif i == 10:
                p = re.compile('[0-9].[0-9]*e\+?-?[0-9]*')
                matches = p.findall(line)
                print(matches)
                a_perp.append([float(x) for x in matches])
            elif i>10: break



    Ok_list.append(Ok)

  #  print(f'Ok: {Ok}, alpha_para_wt:{alpha_para_mean_wt}\n\t alpha: {alpha_para_mean}\n')

fig, ((ax1, ax2),(ax12, ax22),(ax13, ax23)) = plt.subplots(3, 2, sharex=True, figsize=(10, 7))
H = lambda z, Ok, Om=0.31: H0*np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + 1-Ok-Om)
DH_fid = lambda z, Ok: ct.c/1000/H(z, Ok)
n_points = 500
Ok_cont = np.linspace(-0.15,0.15,n_points)
DA_fid = np.array([sp.integrate.quad(DH_fid, 0, zmax, args=(ok,))[0] for ok in Ok_cont])
elinewidth=1
capsize=3
capthick=1.5
for Ok, apara, aperp in zip(Ok_list, a_para, a_perp):
    ax1.errorbar(Ok, apara[0], yerr=apara[1], fmt='xb', 
                 elinewidth=elinewidth, capsize=capsize, capthick=capthick)
    ax2.errorbar(Ok, aperp[0], yerr=aperp[1], fmt='xb', 
                 elinewidth=elinewidth, capsize=capsize, capthick=capthick)
    ax13.errorbar(Ok, DH_fid(zmax, Ok)*apara[0]/rs,  yerr=DH_fid(zmax, Ok)*apara[0]/rs, fmt='xb', 
                 elinewidth=elinewidth, capsize=capsize, capthick=capthick) 
    ax23.errorbar(Ok, DA_fid[-1+int(n_points*(Ok+0.15)/0.3)]*aperp[0]/rs,  yerr=DA_fid[-1+int(100*(Ok+0.15)/0.3)]*aperp[0]/rs, fmt='xb', 
                 elinewidth=elinewidth, capsize=capsize, capthick=capthick) 


ax12.plot(Ok_cont, DH_fid(zmax, Ok_cont)/rs)
ax22.plot(Ok_cont, DA_fid/rs)
ax1.set_ylabel(r'$\alpha_{para}$'), ax2.set_ylabel(r'$\alpha_{perp}$')
ax12.set_ylabel(r'$\left[ DH/r_s\right]_{fid}$'), ax22.set_ylabel(r'$\left[ DA/r_s\right]_{fid}$')
ax13.set_ylabel(r'$DH/r_s$'), ax23.set_ylabel(r'$DA/r_s$')
ax13.set_xlabel(r'$\Omega_k$'), ax23.set_xlabel(r'$\Omega_k$')
plt.tight_layout()
plt.savefig('/home/santi/TFG/figs/DA_DH_flat.pdf')
plt.show()


