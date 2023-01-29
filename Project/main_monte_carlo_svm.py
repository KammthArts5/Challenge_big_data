import numpy as np
import pandas as pd
from def_monte_carlo_svm_functions import *
from def_ga_svm_functions import *
from main_ga_svm import *


#%%

noises = generate_gaussian_noises(5, X_dev, noise_rate=20.5)

Accuracies = store_accuracy(optimal_model, X_dev, y_dev, noises)

print(np.mean(Accuracies))