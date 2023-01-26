import numpy as np
import pandas as pd
from def_regression import *
from def_preprocess_data import *
from sklearn.model_selection import train_test_split


#%% import data
data = import_excel_data("factory_process_KC12 KC12_cleaned.xlsx")
feature_name = data.columns
X = data.iloc[:,:-2]
Y = data.iloc[:,-2:]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.01, random_state = 42)

Y_train_2, Y_train_12 = Y_train.iloc[:,0], Y_train.iloc[:,1]
Y_test_2, Y_test_12 = Y_test.iloc[:,0], Y_test.iloc[:,1]

#%% Defining Logistic regression model

model2 = MLP_model(X_train, Y_train_2)

#coefficient_information(model2, X_train)

model_evalutation(model2, X_test, Y_test_2)

#labels_yep = [str(i) for i in model2.classes_] 


#%% Shap analysis

shap_plot(model2, X_test, X_train,feature_name)