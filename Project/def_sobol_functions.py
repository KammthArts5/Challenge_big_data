import numpy as np
import pandas as pd
from def_monte_carlo_svm_functions import *
from def_ga_svm_functions import *
#%%

def sobol(model, N, parameter, X_dev, y_dev):
    
    std_dev = standards_input(X_dev)
    
    scaling = MinMaxScaler()
    
    noise = generate_gaussian_noises(1000, X_dev, noise_rate=2.5)
    
    noise1 = generate_gaussian_noises(1000, X_dev, noise_rate=2.5)
    noise2 = generate_gaussian_noises(1000, X_dev, noise_rate=2.5)
    
    YaYk = []
    acc_drop = []
    
    for i in range(N):
        
        X_dev_noise = X_dev + noise[i]
        
    
        Ak = X_dev + noise1[i]
        Bk = X_dev + noise2[i]
        
        Ck = Bk
        Ck.replace(Ck[parameter], Ak[parameter]) 
        
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
        