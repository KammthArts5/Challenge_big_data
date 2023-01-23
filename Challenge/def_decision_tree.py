import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix

#%%
def dt_model(X,y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    labels = [str(i) for i in clf.classes_]
    return clf, labels

#%% plot tree

def save_plot_dt(model,feature_name,labels, file_name="model"):
    fig = plt.figure(figsize=(32,18))
    
    tree.plot_tree(model,feature_names=feature_name,class_names=labels, filled=True)
    plt.title("Decision tree " + file_name, size = 40)
    #plt.show()
    fig.savefig("Figures/"+ file_name +".png")

#%% Confusion matrix

def conf_matrix(model,X_test,y_test):
    acc = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    cm = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
                 columns=["Positif réel", "Négatif réel"], index= ["Positif prédit", "Négatif prédit"])
    print("Matrice de confusion :\n", cm)
    print("\nLa présicion de prédiction du modèle est de : {:.2%}".format(acc))
    
    return cm, acc