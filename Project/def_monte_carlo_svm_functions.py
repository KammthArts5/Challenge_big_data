import numpy as np
import pandas as pd
from def_ga_svm_functions import *

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
    return noised_set

def store_accuracy(model, X_dev, y_dev, noises):
    Accuracies = []
    for i in range(len(noises)):
        noised_set = noise_set(X_dev, i, noises)
        Accuracies.append(svm_prediction_acc(model, noised_set, y_dev))
    return np.array(Accuracies)

#%% test

noises = generate_gaussian_noises(1000, len(X_train_scaled[0,:]))

Accuracies = store_accuracy(model, X_dev_scaled, y_dev, noises)