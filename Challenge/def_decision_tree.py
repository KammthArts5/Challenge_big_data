import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import confusion_matrix

#%%
def dt_model(X,y):
    """
    Generate a decision tree model and it extract its labels

    Parameters
    ----------
    X : Pandas DataFrame
        Inputs train data.
    y : Pandas DataFrame
        Output train data.

    Returns
    -------
    clf : Scikit learn Decision Tree
        Decision tree classifier model.
    labels : list
        labels.

    """
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    labels = [str(i) for i in clf.classes_]
    return clf, labels

#%% plot tree

def save_plot_dt(model,feature_name,labels, file_name="model"):
    """
    Save the plot tree in a PNG file

    Parameters
    ----------
    model : Scikit learn Decision tree
        Decision tree to plot.
    feature_name : Pandas Indexes
        Name of the features of the data.
    labels : List
        Labels.
    file_name : String, optional
        Name of the PNG file. The default is "model".

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(32,18))
    
    tree.plot_tree(model,feature_names=feature_name,class_names=labels, filled=True)
    plt.title("Decision tree " + file_name, size = 40)
    #plt.show()
    fig.savefig("Figures/"+ file_name +".png")

#%% Confusion matrix

def conf_matrix(model,X_test,y_test):
    """
    Assess the accuracy of the Decision tree model.

    Parameters
    ----------
    model : Scikit learn Decision Tree
        Decision tree to evaluate.
    X_test : Pandas DataFrame
        Inputs test data.
    y_test : Pandas DataFrame
        Output test data.

    Returns
    -------
    cm : TYPE
        DESCRIPTION.
    acc : TYPE
        DESCRIPTION.

    """
    acc = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    cm = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
                 columns=["Positif réel", "Négatif réel"], index= ["Positif prédit", "Négatif prédit"])
    print("Matrice de confusion :\n", cm)
    print("\nLa présicion de prédiction du modèle est de : {:.2%}".format(acc))
    
    return cm, acc

