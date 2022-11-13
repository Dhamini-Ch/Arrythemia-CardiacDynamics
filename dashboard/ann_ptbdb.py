#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import percentile
from numpy.random import rand
import streamlit as st


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Loading the ptbdb dats set
abnormal     = pd.read_csv("ptbdb_abnormal.csv", header = None)
normal       = pd.read_csv("ptbdb_normal.csv", header = None)

abnormal.shape

normal.shape

abnormal[187].unique()
normal[187].unique()
#Combining abnormal and normal
ptbdb_data = pd.merge(abnormal, normal, how='outer')
ptbdb_data


ptbdb_data_lbl0 = resample(ptbdb_data[ptbdb_data[187]==0], replace=True, n_samples=15000, random_state=113)
ptbdb_data_lbl1 = resample(ptbdb_data[ptbdb_data[187]==1], replace=True, n_samples=15000, random_state=113)


train= pd.concat([ptbdb_data_lbl0, ptbdb_data_lbl1])
labels = train[187].astype('int64')   # last column has the labels

print("Count in each label: ")
print(labels.value_counts())

#Feature Scaling
x = ptbdb_data.iloc[:, :-1].values
y = ptbdb_data.iloc[:, -1].values

#Splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

#Feature Scaling

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ann

accuracy = pd.DataFrame( columns=["Accuracy","Precision","Recall"])
#predictions = np.zeros(shape=(10000,7))
row_index = 0
for i in range(7):
        # bootstrap sampling  
        boot_train = resample(x_train,y_train,replace=True, n_samples=40000, random_state=None)
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=x_train[0].shape),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
  
        # compile the model
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        # Train the model
        model.fit(x_train,y_train,epochs=5,batch_size=32, validation_data=(x_test, y_test) )
        # evaluate the model
        model.evaluate(x_test, y_test)
        import numpy 
cvscores1 = []
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cvscores1.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores1), numpy.std(cvscores1)))
      
      
