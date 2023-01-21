# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:02:41 2022

@author: Wahb
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

#%% - General functions that might be useful

def split_train_dev_test(X, y, Train_size, Dev_size, Test_size):
    """
    Split arrays or matrices into random train, validation and test subsets

    Parameters
    ----------
    X : indexable sequences, e.g. numpy arrays, dataframes, ...
        input parameters.
    y : indexable sequences, e.g. numpy arrays, dataframes, ...
        output parameter.
    Train_size, Dev_size, Test_size : floats
        subsets sizes.
   
    Returns
    -------
    X_train, y_train : indexable sequences
        training set.
    X_dev, y_dev : indexable sequences
        validation set.
    X_test, y_test : indexable sequences
        test set.

    """
    
    assert 0.9999 <= Train_size + Dev_size + Test_size <= 1, "the sum of the subsets sizes must equal 1"
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = Test_size, random_state = 42)
    X_train,X_dev,y_train,y_dev = train_test_split(X_train, y_train, test_size = Dev_size/(1-Test_size), random_state = 42)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def svm_rbf_training(X_train, y_train, C, gamma):
    """
    Train an SVM model with an RBF kernel on training data

    Parameters
    ----------
    X_train, y_train : indexable sequences
        training set.
    C : float
        penalty hyperparameter C.
    gamma : float
        coefficient of rbf kernel.

    Returns
    -------
    model : object
        trained SVM model.

    """
    model = svm.SVC(kernel = 'rbf', C = C, gamma = gamma)
    model.fit(X_train,y_train)
    
    return model
    
def svm_prediction_acc(model, X_dev, y_dev): 
    """
    Evaluate the prediction accuracy of a trained SVM model on new data.

    Parameters
    ----------
    model : object
        trained SVM model.
    X_dev, y_dev : indexable sequences
        data to predict, e.g. validation set, test set, ...
    y_dev : TYPE
        DESCRIPTION.

    Returns
    -------
    float
        prediction accuracy.

    """
    return model.score(X_dev, y_dev)

def scale_data(X):
    """
    Scale a set of data

    Parameters
    ----------
    X : indexable sequence
        Set of input data

    Returns
    -------
    X_scaled : indexable sequence
        Set of input data scaled using SciKit Learn MinMaxScaler

    """
    scaling = MinMaxScaler()             
    scaling.fit(X)
    X_scaled = scaling.transform(X)
    return X_scaled
#%% - Genetic algorithm functions

def init_pop(Size = 100):
    ''' Generates a random population [from uniform distributions] of encoded hyperparameters of SVM RBF kernel.
    
        ============================================================================
        "C" hyperparameter is defined by the first 3 cells of a row; [1, 1000]          
        "Gamma" hyperparameter is defined by the last 4 cells of a row; [0.001, 10]
        ============================================================================
    
        Parameters
        ----------
        Size : size of the population, by default 100.'''
        
    assert type(Size) == int, "Size must be an integer"   
     
    return np.random.randint(10, size=(Size, 7))

def fitness_pop(population, X_train, y_train, X_dev, y_dev):
    '''
    Evaluate the accuracy of every individual of the population

    Parameters
    ----------
    population : indexable sequence
        
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    X_dev : TYPE
        DESCRIPTION.
    y_dev : TYPE
        DESCRIPTION.

    Returns
    -------
    fitnesses : TYPE
        DESCRIPTION.

    '''
    fitnesses = []
    for chrom in population:
        C = 100*chrom[0]+10*chrom[1]+chrom[2]+1
        gamma = chrom[3]+0.1*chrom[4]+0.01*chrom[5]+0.001*(chrom[6]+1)
        model = svm_rbf_training(X_train, y_train, C=C, gamma=gamma)
        fit=0
        fit = svm_prediction_acc(model, X_dev, y_dev)
        #print(C, "\t", gamma, "\t",fit)
        fitnesses.append(fit)
    
    return fitnesses

#%% Selection functions

def tournaments_parents(population, n_parents = 3):
    
    return np.random.randint(len(population), size=(len(population), n_parents))

def best_parents_indexes(population, fitnesses):
    best_parents_index=[]
    Tournaments = tournaments_parents(population)
    for match in Tournaments:
        best_index = match[0]
        if fitnesses[match[1]]>fitnesses[best_index]:
            best_index = match[1]
        if fitnesses[match[2]]>fitnesses[best_index]:
            best_index = match[2]
        best_parents_index.append(best_index)
    
    return best_parents_index

def selected_generation(population, best_indexes):
    new_population = []
    for i in best_indexes:
        new_population.append(population[i])
    
    return np.array(new_population)

#%% Cross-over functions

def cross_over_childs(parent1, parent2):
    child1, child2 =[], []
    for i in range(len(parent1)):
        flip = np.random.randint(2)
        if flip==1:
            child1.append(parent2[i])
            child2.append(parent1[i])
        else:
            child1.append(parent1[i])
            child2.append(parent2[i])
    
    return np.array(child1), np.array(child2)

def cross_over(population):
    new_population = []
    for i in range(len(population)//2):
        child1, child2 = cross_over_childs(population[2*i,:],population[2*i+1,:])
        new_population.append(child1)
        new_population.append(child2)
        if len(population)%2==1:
            new_population.append(population[-1,:])
   
    return np.array(new_population)

#%% Mutation function

def mutation(individual):
    new_individual =[]
    index_rd = np.random.randint(len(individual))
    for i in range(len(individual)):
        if i==index_rd:
            new_individual.append(np.random.randint(10))
        else:
            new_individual.append(individual[i])
    
    return np.array(new_individual)

def mutation_population(population):
    new_population = []
    for i in range(len(population)):
        new_population.append(mutation(population[i,:]))
    
    return np.array(new_population)

#%% Efficiency test

