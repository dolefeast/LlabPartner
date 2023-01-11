import numpy as np
import pandas as pd
from pathlib import Path
import scipy as sp
from scipy.stats import norm

def select_files(files):
    """
    Input: files
    Returns: file_name

    given a list of file names, allows the user to select the file in which they are interested. returns the name of such file.  the file names list is under the name files. 
    """
    print('\nElige un archivo:')
    for i, file_name in enumerate(files, start=1):
        print(f' [{i}]: \t {file_name}')

    while True:
        try:
            file_index = int(input('Elige el archivo que quieres abrir: '))
            break
        except ValueError:
            print('Eso no fue un número\n')
        except IndexError:
            print('Demasiado grande\n')

    file_name = files[file_index-1]

    print('\nAbriendo {}:\n'.format(file_name))
    return file_name

def open_file(file_name):
    """
    Input: file_name
    Returns: (pd.DataFrame, list)
    returns a Dataframe with the given file_name, and the names of the columns
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i, row in enumerate(lines):
            if r'#' not in row:
                break

    names = lines[i-1][1:].split()
    names_copy = []

    for i, char in enumerate(names):
        try:
            int(char[0]) #The first character should always be 1:foo
        except ValueError: #Means it doesn't have the structure 1:foo
            names_copy[-1] += char #In which case it should be a part of the last element of names
        else:
            names_copy.append(char) #This means it had the structure 1:foo

    df = pd.read_csv(file_name, names=None, sep='\s+', header=None, skiprows=i+1)
    return df, names_copy

def curvature(x, y):
    """
    Calculates the curvature of a 2D curve at any point
    Parameters:
        x: array_like
        y: array_like
    Returns:
        k: ndarray 
            The curvature k(t) of the curve (x(t), y(t)) at any 
    """
    x = np.array(x)
    y = np.array(y)

    vx = np.gradient(x)
    vy = np.gradient(y)

    ax = np.gradient(vx)
    ay = np.gradient(vy)
    
    v = np.linalg.norm([x, y])

    return (ax*vy - ay*vy)/v

def remove_bao(k_in, pk_in):
    # This is copied from the code in https://github.com/brinckmann/montepython_public
    # De-wiggling routine by Mario Ballardini
    # This k range has to contain the BAO features:
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
    R = np.arange(len(k_in))
    r_units = k_in[-1]-k_in[0]
    Xi = []
    for r in R: 
        Xi.append(sp.integrate.trapz(k_in * pk_in * np.sin(k_in*r)/r, k_in))
    return R, np.array(Xi)

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

files = list(Path('../class_public/output').glob('*.dat'))


if __name__ == "__main__":
    for i, file_name in enumerate(files, start=1):
        print(f' [{i}]: \t {file_name}')

    while True:
        try:
            file_index = int(input('Elige el archivo que quieres abrir: '))
            break
        except ValueError:
            print('Eso no fue un número\n')

    file_name = files[file_index-1]

    print('\nAbriendo {}:\n'.format(file_name))
    df = open_file(file_name)
    print(df)
