import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
from classy import Class

cosmo = Class()
cosmo.set({ 'output':'tCl ,pCl ,lCl , mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
cosmo.set({'omega_b':0.022032 , 'omega_cdm':0.12038 , 'h':0.67556 , 'A_s':2.215e-9 , 'n_s':0.9619 , 'tau_reio':0.0925})
cosmo.compute()

kk = np.logspace(-4, np.log10(3), 100000)
Pk = []
for k in kk:
    Pk.append(cosmo.pk(k*cosmo.h(),0.)*cosmo.h()**3)
    
ff = interpolate.interp1d(kk, Pk)
def f(k):
    return ff(k)

rr = np.linspace(10,200,90)
Xi = []
for r in rr:
    def g(k):
        return (k**2*np.sin(k*r)/(k*r)*f(k))
    Xi.append(integrate.quad(g, 10**(-4), 3.)[0]/(2.*np.pi**2))

plt.xscale('log'); plt.yscale('log'); plt.xlim(kk[0],kk[-1])
plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$')
plt.plot(kk,Pk,'b-')
plt.show()

plt.xlim(rr[0],rr[-1])
plt.xlabel(r'$r \,\,\,\, [\mathrm{Mpc}/h]$')
plt.ylabel(r'$r^2Xi(r) \,\,\,\, [\mathrm{Mpc}/h]^2$')
plt.plot(rr,rr**2*Xi,'b-')
plt.savefig('xi.pdf')
plt.show()
exit()
