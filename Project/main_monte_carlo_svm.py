import numpy as np
import pandas as pd
from def_monte_carlo_svm_functions import *
from def_ga_svm_functions import *
from main_ga_svm import *


#%%

noises = generate_gaussian_noises(1000, len(X_train_scaled[0,:]))

Accuracies = store_accuracy(optimal_model, X_dev_scaled, y_dev, noises)

