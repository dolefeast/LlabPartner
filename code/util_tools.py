import numpy as np
import pandas as pd
from pathlib import Path
import scipy as sp
import re
from scipy.stats import norm

def select_files(files):
    """
    Input: files
    Returns: file_name

    given a list of file names, allows the user to select the file in which they are interested. returns the name of such file.  the file names list is under the name files. 
    """
    for i, file_name in enumerate(files, start=1):
        print(f' [{i}]: \t {file_name}')

    while True:
        try:
            file_index = int(input('Choose the files to be opened... '))
            break
        except ValueError:
            print('That input was not valid')
        except IndexError:
            print('Too big of an input!')

    file_name = list(files)[file_index-1]

    print('\nAbriendo {}:\n'.format(file_name))
    return file_name

def open_file(file_name):
    """
    Input: file_name
    Returns: (pd.DataFrame, list)
    returns a Dataframe with the given file_name, and the names of the columns
    """
###    with open(file_name, 'r') as f:
#        lines = f.readlines()
#        for i, row in enumerate(lines):
#            if r'#' not in row:
#                break
#
#    names = lines[i-1][1:].split()
#    names_copy = []

#    for i, char in enumerate(names):
#        try:
#            int(char[0]) #The first character should always be 1:foo
#        except ValueError: #Means it doesn't have the structure 1:foo
#            names_copy[-1] += char #In which case it should be a part of the last element of names
#        else:
#            names_copy.append(char) #This means it had the structure 1:foo

    df = pd.read_csv(file_name, names=None, sep='\s+', header=None, comment='#')
    return df

def many_files(files, openfiles=None):
    """Parameters:
        files: A list of file names to chose the files from
    Returns: 
        df_list: a list of pandas dataframes made from the chosen files
        parameters: a list of the curvature parameters of each file
        
    Stops asking for files when inserting a blank space"""
    assert len(list(files)) != 0
    df_list = []
    param_list = []
    if openfiles != None:
        for file_name in files:
                p = re.compile('Om[0-9]*_OL[0-9]*')
                parameters = p.findall(str(file_name))
                print(f'\tOpening {parameters[0]}...')
                df_list.append(open_file(file_name))
                param_list.append(get_params(parameters[0]))

        return df_list, param_list

    for i, file_name in enumerate(files, start=1):
        print(f' [{i}]: \t {file_name}')
    print('The exception handling is weak. Handle with care!')
    file_index = input('Choose the files to open("all" to add all files): ')
    while file_index != 'all':
        print(file_index != 'all')
        try:
            file_index = int(file_index)
            p = re.compile('Om[0-9]*_OL[0-9]*')
            parameters = p.findall(str(file_name))
            if len(parameters) == 0:
                parameters = '_'
            print(f'\tOpening {parameters[0]}...')
            file_name = files[file_index-1]
            df_list.append(open_file(file_name))
            param_list.append(get_params(parameters[0]))
        except ValueError:
            print('Opened [Om, OL]:', param_list)
            return df_list, param_list
        except UnboundLocalError:
            print(f'El archivo {parameters} está vacío')
            continue
#        except IndexError:
#            print(f'The index {file_index} is too big!')
#            continue
        file_index = input('Choose next file to open: ')

def get_params(param_name):
    """
    input:
        param_name: the parameters in the name of each dataframe
    output:
        [Om, OL]: a tuple containing the parameters in the name
        """
    p = re.compile('[0-9]*')
    params = re.split('\D', param_name)
    OmOL = []
    for i in params:
        try:
            OmOL.append(int(i)/100)
        except ValueError:
            continue
    OmOL.append(round(1 - sum(OmOL), 2))
    return OmOL 

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

def remove_bao(k_in, pk_in):
    # De-wiggling routine by Mario Ballardini

    # This k range has to contain the BAO features:
    k_ref=[2.8e-2, 4.5e-1]

    # Get interpolating function for input P(k) in log-log space:
    _interp_pk = sp.interpolate.interp1d( np.log(k_in), np.log(pk_in),
                                             kind='quadratic', bounds_error=False )
    interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))

    # Spline all (log-log) points outside k_ref range:
    idxs = np.where(np.logical_or(k_in <= k_ref[0], k_in >= k_ref[1]))
    _pk_smooth = sp.interpolate.UnivariateSpline( np.log(k_in[idxs]),
                                                     np.log(pk_in[idxs]), k=3, s=0 )
    pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

    # Find second derivative of each spline:
    fwiggle = sp.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
    derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
    d2 = sp.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0)

    # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
    # low-order spline through zeros to subtract smooth trend from wiggles fn.
    wzeros = d2.roots()
    wzeros = wzeros[np.where(np.logical_and(wzeros >= k_ref[0], wzeros <= k_ref[1]))]
    wzeros = np.concatenate((wzeros, [k_ref[1],]))
    wtrend = sp.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)

    # Construct smooth no-BAO:
    idxs = np.where(np.logical_and(k_in > k_ref[0], k_in < k_ref[1]))
    pk_nobao = pk_smooth(k_in)
    pk_nobao[idxs] *= wtrend(k_in[idxs])

    # Construct interpolating functions:
    ipk = sp.interpolate.interp1d( k_in, pk_nobao, kind='linear',
                                      bounds_error=False, fill_value=0. )

    pk_nobao = ipk(k_in)

    return pk_nobao

def calculate_olin(k_in, pk_in):
    """
    Parameters
        a pandas dataframe with 2 columns, the k_in and pk_in
    Output
        a pandas dataframe with 2 columns, the k_in and the pk_in/pk_nobao
    """
    pk_nobao = remove_bao(k_in, pk_in)
    olin = pk_in/pk_nobao
    return olin
def remove_bao_substraction(k_in, pk_in):
    """
    Parameters:
        k_in array
        pk_in array, of same length
    Returns:
        pk_nobao: an array of equal length without the BAO.
    """
    # This is copied from the code in https://github.com/brinckmann/montepython_public
    # De-wiggling routine by Mario Ballardini
    # This k range has to contain the BAO features:
    # changed it to regular substraction.
    k_ref=[2.8e-2, 4.5e-1]

    # Get interpolating function for input P(k) in log-log space:
    _interp_pk = sp.interpolate.interp1d(np.log(k_in), np.log(pk_in),
                                             kind='quadratic', bounds_error=False)
    interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))

    # Spline all (log-log) points outside k_ref range:
    idxs = np.where(np.logical_or(k_in <= k_ref[0], k_in >= k_ref[1]))
    _pk_smooth = sp.interpolate.UnivariateSpline(np.log(k_in[idxs]),
                                                     np.log(pk_in[idxs]), k=3, s=0)
    pk_smooth = lambda x: _pk_smooth(np.log(x)) #Used to be an np.exp around everything
                                                #without it, looks more promising

    # Find second derivative of each spline:
    fwiggle = sp.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
    derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
    d2 = sp.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0)

    # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
    # low-order spline through zeros to subtract smooth trend from wiggles fn.
    wzeros = d2.roots()
    wzeros = wzeros[np.where(np.logical_and(wzeros >= k_ref[0], wzeros <= k_ref[1]))]
    wzeros = np.concatenate((wzeros, [k_ref[1],]))
    wtrend = sp.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)

    # Construct smooth no-BAO:
    idxs = np.where(np.logical_and(k_in > k_ref[0], k_in < k_ref[1]))
    pk_nobao = pk_smooth(k_in)
    pk_nobao[idxs] *= wtrend(k_in[idxs])

    # Construct interpolating functions:
    ipk = sp.interpolate.interp1d(k_in, pk_nobao, kind='linear',
                                      bounds_error=False, fill_value=0.)

    pk_nobao = ipk(k_in)
    return pk_nobao

def fft_bao(k_in, pk_in):
    """Returns the k²P(k)sin(kx)/(kx) integral for given k and P(k) functions
    input: k_in (array), pk_in(array)
    returns: (R, Xi), both np.arrays. Xi is the integral solved for every r
    """
    #Done on a discrete array
    R = np.arange(len(k_in))
    Xi = []
    for r in R:
        Xi.append(sp.integrate.trapz(k_in * pk_in * np.sin(k_in*r)/r, k_in))
    return R, np.array(Xi)

def interpolate_fft(k_in, pk_in, R, kind='linear', n_points=10000):
    """
    Parameters:
        k_in(array): input of k values
        pk_in(array): input of P(k) values
        R: array to map the R values to
    Returns:
        integrate: An integral of k²P(k)sin(kx)/kx from 0 to inf,
                    interpolating P(k) to a continuous function.
    """
    kmin, kmax = k_in[0], k_in[-1]
    pk_interpolate = sp.interpolate.interp1d(k_in, pk_in, kind='linear',
                                            bounds_error=False, fill_value=0.)
    #k_interpolate = sp.interpolate.interp1d(k_in, k_in, kind='linear',
#                                            bounds_error=False, fill_value=0.)
    k_interpolate = np.linspace(kmin, kmax, n_points)
    Xi = []
    for r in R:
       Xi.append(sp.integrate.trapz(k_interpolate * pk_interpolate(k_interpolate) * np.sin(k_interpolate * r)/r, k_interpolate))
    return R, Xi

def interpolate(k_in, pk_in):
    pk_interpolate = sp.interpolate.interp1d(k_in, pk_in, kind='linear',
                                            bounds_error=False, fill_value=0.)
    return pk_interpolate

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x-mean)/4/stddev)**2)

def peaks_fit(k_in, pk_in, function=gaussian, n_points=150):
    """
    Parameters:
        k_in: array
        pk_in: array
        function: the function to be fit to 
        n_points: the points in every split
    Returns:
        gauss_split: np.array that splits the pk_in into 
                     the peaks and fits them to gaussian
    """
    print('Fitting data to the given function...')
    index, _ = sp.signal.find_peaks(-pk_in) #Find minima
    kminima = k_in[index]
    Pminima = pk_in[index]
    #https://stackoverflow.com/questions/44480137/how-can-i-fit-a-gaussian-curve-in-python

    ksplits = np.split(k_in, index) #splits the data with every minima
    pksplits = np.split(pk_in, index)
    kfit = []
    pkfit = []
    for kdata, data in zip(ksplits, pksplits):
        if len(data) <= 15: #The data split is too short
            #pkfit.append(data) #Append the data without doing anything
            continue
        amputation = np.where(np.logical_and(data>=data[0], data>=data[-1]))
        theo_kdata = np.linspace(kdata[0], kdata[-1], n_points) 
        kdata = kdata[amputation]
        data = data[amputation]
        #Estimation of the optimal parameters
        stddev0 = (kdata[0] - kdata[-1])/2
        amplitude0 = max(data)
        mean0 = (kdata[0] + kdata[-1])/2

        data_shift = data - min(data) #Need to shift the data to fit it 
        try:
            optimal, _ = sp.optimize.curve_fit(function, kdata, 
                    data_shift, maxfev=100000)
        except TypeError:
            print('Data array too short!')
            continue
        except RuntimeError:
            print('Divergence!')
            continue
        theo_data = function(theo_kdata, *optimal) + min(data)
        #kfit = np.concatenate((kfit, theo_kdata))
        #pkfit = np.concatenate((pkfit, theo_kdata))
        kfit.append(theo_kdata)
        pkfit.append(theo_data)

    print('Fit done!\n')

    return kfit, pkfit
    return np.concatenate(kfit), np.concatenate(pkfit)

rustico_path = list(Path('/home/santi/TFG/DATA/rustico_output').glob('Power_*'))
class_output = list(Path('/home/santi/TFG/class_public/output').glob('*.dat'))
model = list(Path('/home/santi/TFG/lrg_eboss/model/').glob('*'))
#mcmc_output = list(Path('/home/santi/TFG/lrg_eboss/output').glob('mcmc*.txt'))
mcmc_output = []
for file in Path('/home/santi/TFG/lrg_eboss/output').glob('mcmc*.txt'):
    if '__' in str(file):
        continue
    mcmc_output.append(file)

if __name__ == "__main__":
    print("You are not running the file you should be running!")
