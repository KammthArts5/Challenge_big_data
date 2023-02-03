import numpy as np
import pandas as pd
from def_monte_carlo_svm_functions import *
from def_ga_svm_functions import *
from main_ga_svm import *
#%%

def sobol(model, N, parameter, X_dev, y_dev):
    
    std_dev = standards_input(X_dev)
    
    scaling = MinMaxScaler()
    
    gaussian_noise = generate_gaussian_noises(1000, X_test, noise_rate=2.5)
    
    gaussian_noise1 = generate_gaussian_noises(1000, X_test, noise_rate=2.5) #generation of 2 gaussian noises
    gaussian_noise2 = generate_gaussian_noises(1000, X_test, noise_rate=2.5)
    
    YaYk = []
    acc_drop = []
    
    for i in range(N):
        
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
        