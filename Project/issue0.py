#%% Import library
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

#%% Framework for a simple case
DataFrame = pd.read_excel(r'4vs8.xlsx')

#Data split
X = DataFrame.iloc[:,:-1]
y = DataFrame.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)

#Data scaling
scaling = MinMaxScaler()             
scaling.fit(X_train)
X_train_scaled = scaling.transform(X_train)
X_test_scaled = scaling.transform(X_test)

#Convert dataframe to numpy array
array  = DataFrame.to_numpy()

#Model training - non linear case with RBF kernel
model = svm.SVC(kernel='rbf', C=10, gamma=0.1, decision_function_shape='ovo')
model.fit(X_train, y_train)

#Prediction accuracy of a test set
acc = model.score(X_test, y_test)

#%% Testing best parameters
DataFrame = pd.read_excel(r'4vs8.xlsx')

#Data split
X = DataFrame.iloc[:,:-1]
y = DataFrame.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)
results=np.zeros((len(range(-5,16,2)),len(range(-5,4,2))))

#Data scaling
scaling = MinMaxScaler()             
scaling.fit(X_train)
X_train_scaled = scaling.transform(X_train)
X_test_scaled = scaling.transform(X_test)

#Convert dataframe to numpy array
acc_list = []
for Ci in range (20):
    for gammai in range(18):
        model = svm.SVC(kernel='rbf', C=2**(Ci-5), gamma=2**(gammai-15), decision_function_shape='ovo')
        model.fit(X_train, y_train)
        acc_list.append([Ci,gammai,model.score(X_test, y_test)])


#%%test
df = pd.DataFrame(acc_list)
writer = pd.ExcelWriter('results_issue0.xlsx', engine='xlsxwriter')

df.to_excel(writer, sheet_name='données initiales')