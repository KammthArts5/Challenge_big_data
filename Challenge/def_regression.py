import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap as sh

from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPRegressor

#%%

def MLP_model(X,Y):
    """
    Generate a Multi-layer Perceptron regressor model.

    Parameters
    ----------
    X : Pandas DataFrame
        Inputs train data.
    Y : Pandas DataFrame
        Output train data.

    Returns
    -------
    model : Scikit learn MLP Regressor model
        MLP regressor model.

    """
    model = MLPRegressor(random_state=1 ,max_iter=3000)
    model.fit(X,Y)
    
    return model

def coefficient_information(model,X):
    """
    Display coefficient information of the model

    Parameters
    ----------
    model : Scikit learn MLP Regressor model
        Model to evaluate.
    X : Pandas DataFrame
        Input data.

    Returns
    -------
    None.

    """
    temp=pd.DataFrame(model.coef_[0],index=X.columns,columns=['coef'])
    print(temp)
    
def model_evalutation(model,X_test,Y_test):
    
    return model.score(X_test,Y_test)
    

    
    
#%% Shap analysis

def shap_plot(model, X_test,X_train, heads):
    """
    Display a bar plot of the shapley value of the model.

    Parameters
    ----------
    model : Scikit learn MLP Regressor model
        Model to evaluate.
    X_test : Pandas DataFrame
        Input test data.
    X_train : Pandas DataFrame
        Input train data.
    heads : List
        List of the parameters label.

    Returns
    -------
    None.

    """
    explainer = sh.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    sh.summary_plot(shap_values, heads, plot_type="bar")
    
    