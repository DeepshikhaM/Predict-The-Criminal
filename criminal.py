# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:53:50 2019

@author: DEEPSHIKHA
"""
import tensorflow
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data =  pd.read_csv('criminal_train.csv')

data1 = pd.read_csv('criminal_test.csv')


#cols = data.columns
#
#l = []
#
#for i in cols:
#    l.append(i)
    
    
X = data.iloc[:, 1:-1].values
Y = data.iloc[:, -1].values

data.shape
data1.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

print(X)

#from sklearn.model_selection import train_test_split
#X_train , X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 1/5, random_state = 42)

classifier = Sequential()

classifier.add(Dense(units = 50, activation='sigmoid', input_dim = 70))

classifier.add(Dense(units = 50, activation='sigmoid'))

classifier.add(Dense(units = 50, activation='sigmoid'))

classifier.add(Dense(units = 1, activation ='sigmoid'))

######## COMPILE A MODEL#########################
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X , Y, batch_size = 42, epochs =20 )


#y_pred = np.round(classifier.predict(X_test))
#df = pd.DataFrame(y_pred)
#
#df[0].value_counts()

te = data1.iloc[: , 1:].values
te

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
te = sc.fit_transform(te)

print(te) 
y_pr = np.round(classifier.predict(te))
y_pr = pd.DataFrame(y_pr)

y_pr.to_csv('Kuch bhi.csv')

pr = np.round(classifier.predict(te))
y_pr = pd.DataFrame(y_pr)

y_pr.to_csv('Kuch bhi.csv')



############################################################################################

model = Sequential()
model.add(Dense(units = 40, activation='sigmoid', input_dim = 70))
model.add(Dense(units = 40, activation='relu'))
model.add(Dense(units = 40, activation='softmax'))
model.add(Dense(units = 40, activation='softmax'))
model.add(Dense(units = 32, activation='relu'))

model.add(Dense(units = 1, activation ='relu'))

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X , Y, batch_size = 42, epochs = 20 )

te = data1.iloc[: , 1:].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
te = sc.fit_transform(te)
 
y_pr = np.round(classifier.predict(te))
y_pr = pd.DataFrame(y_pr)

y_pr.to_csv('Secondcrimm.csv')





