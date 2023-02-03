import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
#%% - General functions that might be useful

def split_train_dev_test(X, y, Train_size, Dev_size, Test_size):
    """
    Split arrays or matrices into random train, validation and test subsets
    Parameters
    ----------
    X : indexable sequences, e.g. numpy arrays, dataframes, ...
        input parameters.
    y : indexable sequences, e.g. numpy arrays, dataframes, ...
        output parameter.
    Train_size, Dev_size, Test_size : floats
        subsets sizes.
   
    Returns
    -------
    X_train, y_train : indexable sequences
        training set.
    X_dev, y_dev : indexable sequences
        validation set.
    X_test, y_test : indexable sequences
        test set.
    """
    
    assert 0.9999 <= Train_size + Dev_size + Test_size <= 1, "the sum of the subsets sizes must equal 1"
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = Test_size, random_state = 42)
    X_train,X_dev,y_train,y_dev = train_test_split(X_train, y_train, test_size = Dev_size/(1-Test_size), random_state = 42)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def svm_rbf_training(X_train, y_train, C, gamma):
    """
    Train an SVM model with an RBF kernel on training data
    Parameters
    ----------
    X_train, y_train : indexable sequences
        training set.
    C : float
        penalty hyperparameter C.
    gamma : float
        coefficient of rbf kernel.
    Returns
    -------
    model : object
        trained SVM model.
    """
    model = svm.SVC(kernel = 'rbf', C = C, gamma = gamma)
    model.fit(X_train,y_train)
    
    return model
    
def svm_prediction_acc(model, X_dev, y_dev): 
    """
    Evaluate the prediction accuracy of a trained SVM model on new data.
    Parameters
    ----------
    model : object
        trained SVM model.
    X_dev, y_dev : indexable sequences
        data to predict, e.g. validation set, test set, ...
    y_dev : TYPE
        DESCRIPTION.
    Returns
    -------
    float
        prediction accuracy.
    """
    return model.score(X_dev, y_dev)

def scale_data(X):
    """
    Scale a set of data
    Parameters
    ----------
    X : indexable sequence
        Set of input data
    Returns
    -------
    X_scaled : indexable sequence
        Set of input data scaled using SciKit Learn MinMaxScaler
    """
    scaling = MinMaxScaler()             
    scaling.fit(X)
    X_scaled = scaling.transform(X)
    return X_scaled

#%%Get data


def get_data():
    data = pd.read_excel(r'C:\Users\erwan\OneDrive\Bureau\Mes dossiers\TEMP\4vs8.xlsx')    
    X = data.iloc[:,:-1]; y = data.iloc[:,-1]                                                 
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, 0.8, 0.1, 0.1)
    scaling = MinMaxScaler()             
    scaling.fit(X_train)
    X_train_scaled = scaling.transform(X_train)
    X_dev_scaled = scaling.transform(X_dev)
    X_test_scaled = scaling.transform(X_test)

    
    return X_train_scaled, X_dev_scaled, X_test_scaled, X_train, X_dev, X_test, y_train, y_dev, y_test



def get_std(X):
    
    
    #standard deviation of each parameter
    std = []
    
    for column in X :
        
        std.append(X[column].std())
    

    return std
#%%
print(get_std())

#%%Gaussian noise : generation of a matrix of noises. The noise is set at 5% to observe something.

def gen_gaussian_noise(X, N): 
    
    gaussian_noise = []
    
    std = get_std(X)
    
    for i in range(N):
    
        gaussian_noise.append(np.array([[np.random.normal(0, std[i]*0.05) for i in range(X.shape[1])] for j in range(X.shape[0])]))
    
    return gaussian_noise

    

#%% acc_ loss returns the accuracy drop between one noisy set and a witness one

def acc_loss(N):
    
    X_train_scaled, X_dev_scaled, X_test_scaled, X_train, X_dev, X_test, y_train, y_dev, y_test = get_data()
    
    std_dev = get_std(X_dev)
    
    acc_drop = []
    
    scaling = MinMaxScaler()
    
    gaussian_noise = gen_gaussian_noise(X_dev, N)
    model = svm.SVC(kernel = 'rbf', C = 215, gamma = 6.035)
    model.fit(X_train_scaled,y_train)
    
    for i in tqdm(range(N), desc = 'tqdm() Progress Bar' ):
        
        X_dev_noise = X_dev + gaussian_noise[i]
        scaling.fit(X_dev_noise)
        X_dev_noise = scaling.transform(X_dev_noise)
        
        acc_drop.append(model.score(X_dev_scaled, y_dev)-model.score(X_dev_noise, y_dev))
        
        
        
    return acc_drop
    

#%%
acc_drop = acc_loss(1000)
print(np.mean(acc_drop), np.std(acc_drop))

#%% 
for i in range(5):
    print(np.random.normal(0,2))
#%%Sobol analysis

def sobol(N,parameter):
    
    
    X_train_scaled, X_dev_scaled, X_test_scaled, X_train, X_dev, X_test, y_train, y_dev, y_test = get_data()
    
    std_dev = get_std(X_dev)
    
    scaling = MinMaxScaler()
    
    gaussian_noise = gen_gaussian_noise(X_dev, N)
    model = svm.SVC(kernel = 'rbf', C = 215, gamma = 6.035)
    model.fit(X_train_scaled,y_train)
    
    gaussian_noise1 = gen_gaussian_noise(X_dev, N) #generation of 2 gaussian noises
    gaussian_noise2 = gen_gaussian_noise(X_dev, N)
    
    YaYk = []
    acc_drop = []
    
    for i in tqdm(range(N), desc = 'tqdm() Progress Bar' ):
        
        X_dev_noise = X_dev + gaussian_noise[i]
        
    
        Ak = X_dev + gaussian_noise1[i]  #2 Gaussian Noises for the noisy sets Ak and Bk
        Bk = X_dev + gaussian_noise2[i]
        
        Ck = Bk
        Ck.replace(Ck[parameter], Ak[parameter])  #Only the column of the wanted parameter of Ak is replaced in the set Ck
        
        scaling.fit(Ak)
        Ak_scaled = scaling.transform(Ak)
        scaling.fit(Ck)
        Ck_scaled = scaling.transform(Ck)
        
        YaYk.append(model.score(Ak_scaled,y_dev)*model.score(Ak_scaled,y_dev))
        acc_drop.append(model.score(Ak_scaled,y_dev)-model.score(Ak_scaled,y_dev))
    v = []   
    for i in range(len(YaYk)):
        v.append(YaYk[i]**2)
    
    V = np.mean(v)-(np.mean(acc_drop))**2
        
    return (np.mean(YaYk)-(np.mean(acc_drop))**2)/V
        
        
#%%
#This scripts executes the codes above and returns the Sobol indices (Si and Sti) for all the parameters

X_train_scaled, X_dev_scaled, X_test_scaled, X_train, X_dev, X_test, y_train, y_dev, y_test = get_data()

Si = []
Sti = []
for parameter in X_dev :
    s = sobol(100,parameter)
    Si.append(s)
    Sti.append(1-s) #Not really sure of this one. It is described as such in the course but not in the literature.
print(Si, Sti)
    














