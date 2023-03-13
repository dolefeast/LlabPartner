#Main parameters
#Type of fit (BAOISO/BAOANISO/FS/FSBAOISO/FSBAOANISO/FSalphasrecon): BAOANISO

#Include power spectrum (yes/no): yes
#FIT BAO to P0/P2/P4/P02/P024/P24/P04: P02
#k-range for P0 computation (double/double): 0.02 0.30
#k-range for P2 computation (double/double): 0.02 0.30
#k-range for P4 computation (double/double): 0.02 0.30
#FIT FS to P0/P2/P4/P02/P024/P24/P04: P024
#k-range for P0 computation (double/double): 0.02 0.15
#k-range for P2 computation (double/double): 0.02 0.15
#k-range for P4 computation (double/double): 0.02 0.15
#Number of chunks: 2

#Read input parameters for Power Spectrum
#Path to data1BAO: /home/ssanz/TFG/outputs_santi/rustico/Power_Spectrum_eBOSS_NGC_Om031_OL069.txt
#Path to data1BAO: /home/ssanz/TFG/outputs_santi/rustico/Power_Spectrum_eBOSS_NGC_Om031_OL069.txt
#Path to data1FS: none
#Path to data2FS: none
#Path to mocks1BAO/cov1: /home/ssanz/TFG/lrg_eboss/mocks/Power_Spectrum_comb_NGC_ezmocks_
#Path to mocks2BAO/cov1: /home/ssanz/TFG/lrg_eboss/mocks/Power_Spectrum_comb_SGC_ezmocks_
#Path to mocks1FS: none
#Path to mocks2FS: none
#number of realizations: 1000
#Path to mask1 (none if no mask): /home/ssanz/TFG/lrg_eboss/window/W_randomdata_eBOSS_LRGpCMASS_NGC_v72_yama_newformat.txt
#Path to mask2 (none if no mask): /home/ssanz/TFG/lrg_eboss/window/W_randomdata_eBOSS_LRGpCMASS_SGC_v72_yama_newformat.txt
#Renormalize window (yes/no): yes
#Apply mask as a matrix (yes/no): yes
#Correction by (Hartlap/Sellentin-Heavens/none): Hartlap
#Covariance for FSalpharecon (fixed/varying): varying

#Read input parameters for Bispectrum
#Include bispectrum (yes/no): no
#Fit bispectrum (B) or reduced bispectrum (Q): B
#Path to data1BAO: /home/ssanz/TFG/outputs_santi/rustico/Power_Spectrum_eBOSS_NGC_Om031_OL069.txt
#Path to data1BAO: /home/ssanz/TFG/outputs_santi/rustico/Power_Spectrum_eBOSS_NGC_Om031_OL069.txt
#Path to mocks1BAO: none
#Path to mocks2BAO: none
#k-range for B0 computation (double/double): 0.02 0.30
#Path to data1FS: none
#Path to data2FS: none
#Path to mocks1FS: none
#Path to mocks2FS: none
#k-range for B0 computation (double/double): 0.02 0.15

#Do Compression? (no/linear/nonlinear): no
#Path to compression file: compression_file_format.txt

#BAO Polynomial fit
#Order of polynomial broadband: 5
#Sigma variables (effective/para-perp): para-perp
#Sigma variables independent (yes/no): yes
#f-factor for Sigma dependence: 0.8204
#Smoothing scale in Mpc/h (only for BAOANISO): 0

#Sigma_variables
#SigmaP_0 (free/fixed/gaussian): fixed
#SigmaP_0 (mean/devstd): 5 1
#SigmaP_2 (free/fixed/gaussian): fixed
#SigmaP_2 (mean/devstd): 10 1
#SigmaP_4 (free/fixed/gaussian): fixed
#SigmaP_4 (mean/devstd): 9 3
#SigmaB_0 (free/fixed/gaussian): fixed
#SigmaB_0 (mean/devstd): 9 3
#Sigma_para (free/fixed/gaussian): fixed
#Sigma_para (mean/devstd): 9.4 3
#Sigma_perp (free/fixed/gaussian): fixed
#Sigma_perp (mean/devstd): 4.8 3

#FS priors/options
#Apply noise in the model or subtract it in the data (model/data): data
#Anoise (flat/gaussian): flat
#Anoise (mean/devstd): 1 0.30
#b2 (flat/gaussian): flat
#b2 (mean/devstd): 0 2.5
#b2s2 (flat/gaussian): flat
#b2s2 (mean/devstd): 0 5
#b3nl (flat/gaussian): flat
#b3nl (mean/devstd): 0 5

#Priors on alpha: 0.7 1.3
#Path to smooth Plin: /home/ssanz/TFG/outputs_santi/linspace_class/psmlin__linspace_Om031_OL069.txt
#Path to Olin: /home/ssanz/TFG/outputs_santi/linspace_class/Olin__linspace_Om031_OL069.txt
#Type of fit (mcmc/analytic): mcmc

#For mcmc fit
#Number of threads: 32
#Use proposal covariance (yes/no): yes
#Path to proposal covariance: /home/ssanz/TFG/outputs_santi/rustico_Om031_OL069/class_Om031_OL069/mcmcBAOANISO_output_fid_Om031_OL069.txt
#Maximum number of .c accepted steps: 500000
#mcmc sampling step (recommended 1.9): 1.

#For BAO anaylytical fit
#Interval for alpha: 0.00001

#For FS mcmc fit
#Input file from PTcool: none
#Type of PT-Model (linear/1L-SPT/2L-SPT/1L-RPT/2L-RPT): 2L-RPT
#Type of RSD-model (Kaiser87/Scoccimarro04/TNS10): TNS10
#Type of FoG-model (Exponential/Lorentzian/Exponential_avir): Lorentzian
#Type of model for bispectrum (tree-level/1L-SPT/GilMarin14): GilMarin14
#Lagrangian local bias b2s2 (yes/no/off): yes
#Lagrangian local bias b3nl (yes/no/off): yes
#RSD-fit (yes/no/shape/shape2): yes
#sigma8 free parameter (yes/no): no
#FoG free parameter (yes/no): yes
#Same FoG parameter for bispectrum (yes/no): no

#Read out parameters
#Path to output: /home/ssanz/TFG/outputs_santi/rustico_Om031_OL069/2nd_class_Om031_OL069/
#Identifier of output: 2nd_fid_Om031_OL069
#Plot best-fit (yes/no): yes
