import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap as sh

from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPRegressor

#%%

def logistic_reg_model(X,Y):
    model = LogisticRegression(solver='liblinear',multi_class='auto')
    model.fit(X,Y) 
    
    return model

def MLP_model(X,Y):
    model = MLPRegressor(random_state=1, max_iter=500)
    model.fit(X,Y)
    
    return model

def coefficient_information(model,X):
    temp=pd.DataFrame(model.coef_[0],index=X.columns,columns=['coef'])
    print(temp)
    
def model_evalutation(model,X_test,Y_test):
    
    return model.score(X_test,Y_test)
    

    
    
#%% Shap analysis

def shap_plot(model, X_test,X_train, heads):
    explainer = sh.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    sh.summary_plot(shap_values, heads, plot_type="bar")
    
    