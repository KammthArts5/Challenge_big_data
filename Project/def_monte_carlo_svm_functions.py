import numpy as np
import pandas as pd
from def_ga_svm_functions import *

#%% 

def standards_input(X):
    array_X=np.copy(X)
    sigmas = []
    for i in range(len(array_X[0,:])):
        sigmas.append(np.std(array_X[:,i]))
    return sigmas

def generate_gaussian_noises(N, X, mu=0, noise_rate=2.5):
    """
    Generate a set of N noises on dim dimensions.


    Parameters
    ----------
    N : int
        Number of noises to generate.
    dim : int
        Number of dimensions of the gaussian noise.
    mu : float, optional
        Mean value of the gaussian noise. The default is 0.
    sigma : float, optional
        Noise magnitude
        Standard deviation of the gausssian noise. The default is 0.025.

    Returns
    -------
    Numpy array
        Set of noises.
        
    """
        
    sigmas = standards_input(X)
    noises=[]
    for i in range(N):
        noises.append(np.array([[np.random.normal(0, sigmas[i]*noise_rate/100) for i in range(X.shape[1])] for j in range(X.shape[0])]))
    
    return np.array(noises)

def noise_set(data, i, noises):
    """
    Set the i th noise to a dataset

    Parameters
    ----------
    data : Numpy Array
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
    noised_data = data + noises[i,:]
    return scale_data(noised_data)

def store_accuracy(model, X_dev, y_dev, noises):
    Accuracies = []
    for i in range(len(noises)):
        noised_set = noise_set(X_dev, i, noises)
        Accuracies.append(svm_prediction_acc(model, noised_set, y_dev))
    return np.array(Accuracies)

def acc_loss(model, N, X_dev, X_dev_scaled, y_dev):
    
    std_dev = standards_input(X_dev)
    
    acc_drop = []
    
    gaussian_noise = generate_gaussian_noises(N,X_dev,noise_rate=2.5)
    
    for i in range(N):
        X_dev_noise = X_dev + gaussian_noise[i]
        X_dev_noise = scale_data(X_dev_noise)
        
        acc_drop.append(model.score(X_dev_scaled, y_dev)-model.score(X_dev_noise, y_dev))
        
        
        
    return acc_drop

#%% test

