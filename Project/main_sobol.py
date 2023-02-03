import numpy as np
import pandas as pd
from def_monte_carlo_svm_functions import *
from def_ga_svm_functions import *
from main_ga_svm import *
from def_sobol_functions import *

#%%

noises = generate_gaussian_noises(5, X_dev, noise_rate=20.5)

initial_score = svm_prediction_acc(optimal_model, X_test_scaled, y_test)

accur = store_accuracy(optimal_model, X_dev, y_dev, noises)

Y_accur = store_accuracy(optimal_model, X_dev, y_dev, noises) - initial_score

expected_value = np.mean(Y_accur)

#%%


Si = []
Sti = []
for parameter in X_dev :
    s = sobol(optimal_model,100,parameter, X_dev, y_dev)
    Si.append(s)
    Sti.append(1-s) #Not really sure of this one. It is described as such in the course but not in the literature.
print(Si, Sti)