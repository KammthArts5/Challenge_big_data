import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#%%

def get_metadata(data):
    """
    Parameters
    ----------
    data : pandas dataframe
        jeu de données importé.

    Returns
    -------
    un tuple de trois éléments:
        1- nombre de lignes.\n
        2- nombre de colonnes.\n
        3- les noms des paramètres.\n
    """
    
    return data.shape[0], data.shape[1], data.columns


def normalize_data(data):
    """
    En théorie des probabilités et en statistique, une variable centrée réduite est la transformée d'une variable aléatoire par une application, de telle sorte que sa moyenne soit nulle (𝑥_moy=0) et son écart type égal à un (𝜎=1).
    
    Parameters
    ----------
    data : pandas dataframe
        Jeu de données importé.

    Returns
    -------
    numpy array
        Données centrées réduites.
    """
    
    sc = StandardScaler()
    return sc.fit_transform(data)

def corr_axes_param(val_pr, vec_pr, dim, Columns):
    """
    Give the correlation between the parameters and the principle axes
    
    Parameters
    ----------
    val_pr : numpy array
        valeurs propres.
    vec_pr : numpy array
        vecteurs propres.
    dim : int
        dimension du jeu de données.   
    Columns : séquence(liste, dataframe,...)
        noms des paramètres.

    Returns
    -------
    pandas datframe
        Corrélations entre les paramètres et les axes principaux.

    """
    mat_cor = np.zeros((dim,dim))      # corrélations entre variables(lignes) et axes principaux(colonnes)
    for k in range(dim):
        mat_cor[:,k] = vec_pr[k,:] * val_pr[k]**0.5
    
    return pd.DataFrame(data=mat_cor, index=Columns, columns=["Axe"+str(i+1) for i in range(dim)])