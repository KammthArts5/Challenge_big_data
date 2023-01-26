import numpy as np
import pandas as pd
from def_regression import *
from def_preprocess_data import *
from sklearn.model_selection import train_test_split


#%% import data KC2
data = import_excel_data("factory_process_KC2.xlsx") #Use the right file with the selected parameters
feature_name = data.columns
X = data.iloc[:,:-2]
Y = data.iloc[:,-2:]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)

Y_train_2, Y_train_12 = Y_train.iloc[:,0], Y_train.iloc[:,1]
Y_test_2, Y_test_12 = Y_test.iloc[:,0], Y_test.iloc[:,1]

#%% Defining Logistic regression model KC2

model2 = MLP_model(X_train, Y_train_2)
    
precision=model_evalutation(model2, X_test, Y_test_2)
print(precision)
#labels_yep = [str(i) for i in model2.classes_] 

#%% Shap analysis KC2

# shap_plot(model2, X_test, X_train,feature_name)

#%% import data KC2
data = import_excel_data("factory_process_KC12_test.xlsx") #Use the right file with the selected parameters
feature_name = data.columns
X = data.iloc[:,:-2]
Y = data.iloc[:,-2:]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)

Y_train_2, Y_train_12 = Y_train.iloc[:,0], Y_train.iloc[:,1]
Y_test_2, Y_test_12 = Y_test.iloc[:,0], Y_test.iloc[:,1]


#%% Defining Logistic regression model KC12

model12 = MLP_model(X_train, Y_train_12)

precision=model_evalutation(model12, X_test, Y_test_12)
print(precision)

#%% Shap analysis KC12

shap_plot(model12, X_test, X_train,feature_name)