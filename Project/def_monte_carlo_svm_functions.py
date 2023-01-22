import numpy as np
import pandas as pd


#%% 

def generate_gaussian_noises(N, dim):
    """
    Generate a set of N noises on dim dimensions.

    Parameters
    ----------
    N : int
        Number of noises to generate.
    dim : int
        Number of dimensions of the gaussian noise.

    Returns
    -------
    Numpy array
        Set of noises.

    """
    mu=0
    sigma=0.025
        
    return np.random.normal(mu, sigma,[N,dim])

def noise_set(scaled_data, i, noises):
    """
    Set the i th noise to a set of scaled data

    Parameters
    ----------
    scaled_data : Numpy Array
        Data to noise.
    i : int
        Noise to set.
    noises : Numpy Array
        Noise to set.

    Returns
    -------
    noised_set : Numpy Array
        Noised data.

    """
    noised_set = np.copy(scaled_data)
    for k in range(len(scaled_data)):
        noised_set[k,:] = scaled_data[k,:] + noises[i,:]
        print(noised_set[k,:])
    return noised_set

#%% test

A = np.array([[1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0],
              [1.0,1.0,1.0]])
noises = generate_gaussian_noises(5, 3)

B = noise_set(A, 2, noises)