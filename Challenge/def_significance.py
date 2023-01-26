import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from heatmap import corrplot

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

#%% Plot results

def plot_eigenvalues(val_pr, Inertie_sum, dim, show=False):
    """
    Cette fonction permet de générer une visualisation des différentes valeurs propres et de la variation cumulative.
    
    Parameters
    ----------
    val_pr : numpy array
        valeurs propres.
    Inertie_sum : numpy array
        variation cumulative.
    dim : int
        dimension du jeu de données.        
    """
    
    fig, ax1 = plt.subplots(figsize=(10,6))

    X_labels =  ['λ'+str(i+1) for i in range(dim)]
    ax1.bar(X_labels, height=val_pr, color='green')
    ax1.grid(axis='y')
    plt.xlabel("Composantes pricipales")
    plt.ylabel("Valeurs propres")
    
    ax2=ax1.twinx()       # pour superposer deux grapghes
    ax2.plot(Inertie_sum,color='red')
    for i in range(0,dim):
        ax2.annotate(("{:.2%}".format(Inertie_sum[i])), (i,Inertie_sum[i]), size=12)  # 1er argument = l'annotation; 2eme = le point (xi,yi)
    ax2.scatter(np.arange(0,dim), Inertie_sum, s=20, color='black')
    plt.ylabel("Variation cumulative")
    
    if show:    
        plt.show()
    return fig

def plot_projection(axe_i, axe_j, param_gradient, data, Tr_data, Corr_ax_par, Inertie, Columns, dim, show=False):
    """
    la fonction projection permet de : 
        +) représenter les données dans un plan 2D défini par les deux axes principaux 'axe_i', 'axe_j' et par le gradient de couleur correspondant au paramètre 'param_gradient'.\n
        +) représenter les paramètres d'entrée sur ce plan 2D afin de mettre en exergue les différentes corrélations.

    Parameters
    ----------
    axe_i : int
        1er axe principal à visualiser.
    axe_j : int
        2eme axe principal à visualiser.
    param_gradient : str
        nom du paramètre.
    data : pandas dataframe
        jeu de donnée importé.
    Tr_data : numpy array
        jeu de donnée après transformation dans la nouvelle base.
    Corr_ax_par : pandas dataframe
        matrice de corrélation entre les axes pricipaux et les paramètres.
    Inertie : numpy array
        variations liées aux axes principaux.
    Columns : séquence(liste, dataframe,...)
        noms des paramètres.
    dim : int
        dimension du jeu de données.
    """
    
    axe_i = int(axe_i-1); axe_j = int(axe_j-1)
    assert type(param_gradient) == str, "'param_gradient' doit être une chaîne de caractères."
    
    fig= plt.figure(figsize=(17,8))
    
    ax1 = fig.add_subplot(121)
    ax1.spines['top'].set_color('none')       # 'top' arête sans couleur
    ax1.spines['right'].set_color('none')
    ax1.spines['bottom'].set_position('zero') # positionner 'bottom' arête au niveau du zéro 
    ax1.spines['left'].set_position('zero')
    
    plt.scatter(Tr_data[:,axe_i],Tr_data[:,axe_j],s=7.5, c=data[param_gradient], cmap='Reds')
    plt.title("Plan {}x{}: Inertie {:.2%}".format(axe_i+1, axe_j+1, Inertie[axe_i]+Inertie[axe_j]), size=14)
    plt.grid(linestyle=':')
    
    ax2 = fig.add_subplot(122)
    ax2.spines['top'].set_color('none')       
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('zero') 
    ax2.spines['left'].set_position('zero')
    C = plt.Circle((0,0),1,color='r',fill=False)
    ax2.add_artist(C)
    
    # Définir les flèches
    O=np.zeros((dim,1))
    plt.quiver(O, O, Corr_ax_par.iloc[:,axe_i], Corr_ax_par.iloc[:,axe_j],angles='xy', scale_units='xy', scale=1, color='blue', width = 0.003, headwidth=1)
    ax2.scatter(Corr_ax_par.iloc[:,axe_i], Corr_ax_par.iloc[:,axe_j],s=20,color='red')
    for i in range(0,dim):
        ax2.annotate(Columns[i], (Corr_ax_par.iloc[i,axe_i], Corr_ax_par.iloc[i,axe_j]), size=12, color = 'red')
    
    plt.xlim((-1.05,1.05)); plt.ylim((-1.05,1.05))
    plt.grid(linestyle=':')    
    plt.title("Plan {}x{}: Inertie {:.2%}".format(axe_i+1, axe_j+1, Inertie[axe_i]+Inertie[axe_j]), size=14)
    
    if show:    
        plt.show()
    return fig

def plot_corr_mat(data, show=False):
    """
    Cette fonction permet de générer une visualisation de la matrice de corrélation d'un jeu de données.

    Parameters
    ----------
    data : pandas dataframe
        jeu de donnée importé.

    """
    fig02 = plt.figure(figsize=(12, 12))
    corrplot(data.corr(), size_scale=300, marker="o")
    return fig02, data.corr()