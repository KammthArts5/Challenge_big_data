# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:00:27 2022

@author: Wahb
"""

#%% - libraries import
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from def_ga_svm_functions import *    

#%% - Data import and split
data = pd.read_excel(r'4vs8.xlsx')    # Import data
X = data.iloc[:,:-1]
y = data.iloc[:,-1]                                                 
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, 0.6, 0.2, 0.2)          # Split data

#%% - Data scaling (to be done)
X_train_scaled, X_dev_scaled, X_test_scaled = scale_data(X_train), scale_data(X_dev), scale_data(X_test)

#%% - Example of an application of SVM
# model = svm_rbf_training(X_train_scaled, y_train, C=478, gamma = 5.72)
# acc = svm_prediction_acc(model, X_dev_scaled, y_dev)

#%% - Genetic algorithm : first generation (to be done)
Ngen = 100

pop = init_pop(100)
Generations = [pop]


#- Genetic algorithm : loop (to be done)
new_gen = pop
fit = fitness_pop(new_gen, X_train_scaled, y_train, X_test_scaled, y_test)
for i in range(Ngen):
    indexes = best_parents_indexes(new_gen, fit)
    parents=selected_generation(new_gen, indexes)
    Generations.append(parents)
    childs = cross_over(parents)
    new_gen = mutation_population(childs)
    fit = fitness_pop(new_gen, X_train_scaled, y_train, X_test_scaled, y_test)
    print("Step: ", i, "/", Ngen,"\t Mean accuracy: ", np.mean(fit))
    
#print(new_gen)



#%% Tests

optimal_model= find_best_model(new_gen, X_train_scaled, y_train, X_test_scaled, y_test)
