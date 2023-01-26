import numpy as np
import pandas as pd
from def_decision_tree import *
from def_preprocess_data import *
from sklearn.model_selection import train_test_split


#%% import data
data = import_excel_data("factory_process_Conformity KC2  KC12.xlsx")
feature_name = data.columns
X = data.iloc[:,:-2]
Y = data.iloc[:,-2:]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.6, random_state = 42)

Y_train_2, Y_train_12 = Y_train.iloc[:,0], Y_train.iloc[:,1]
Y_test_2, Y_test_12 = Y_test.iloc[:,0], Y_test.iloc[:,1]

#%% test

model2, labels2 = dt_model(X_train, Y_train_2) #KC2
save_plot_dt(model2,feature_name,labels2,"model2")
conf_matrix(model2, X_test, Y_test_2)

model12, labels12 = dt_model(X_train, Y_train_12) #KC12
save_plot_dt(model12,feature_name,labels2,"model12")
conf_matrix(model12, X_test, Y_test_12)

