#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




train = pd.read_csv("combined_train.csv", header=None)
test = pd.read_csv("combined_test.csv", header=None)

train.shape

test.shape

train[187].unique()

test[187].unique()

train_lbl0 = resample(train[train[187]==0], replace=True, n_samples=15000, random_state=113)
train_lbl1 = resample(train[train[187]==1], replace=True, n_samples=15000, random_state=113)
train_lbl2 = resample(train[train[187]==2], replace=True, n_samples=15000, random_state=113)
train_lbl3 = resample(train[train[187]==3], replace=True, n_samples=15000, random_state=113)
train_lbl4 = resample(train[train[187]==4], replace=True, n_samples=15000, random_state=113)
train_lbl5 = resample(train[train[187]==5], replace=True, n_samples=15000, random_state=113)
train_lbl6 = resample(train[train[187]==6], replace=True, n_samples=15000, random_state=113)

train= pd.concat([train_lbl0, train_lbl1, train_lbl2, train_lbl3, train_lbl4, train_lbl5, train_lbl6])

labels = train[187].astype('int64')   # last column has the labels

print("Count in each label: ")
print(labels.value_counts())

x_train = train.iloc[:, 1:-1].values
y_train = train.iloc[:, -1].values

x_test = test.iloc[:, 1:-1].values
y_test = test.iloc[:, -1].values

accuracy = pd.DataFrame( columns=["Accuracy","Precision","Recall"])
#predictions = np.zeros(shape=(10000,7))
row_index = 0
for i in range(7):
        # bootstrap sampling  
        boot_train = resample(x_train,y_train,replace=True, n_samples=40000, random_state=None)
        model2 = tf.keras.Sequential([
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
            tf.keras.layers.Dense(7, activation=tf.nn.softmax)])
  
        # compile the model
        model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        # Train the model
        model2.fit(x_train,y_train,epochs=5,batch_size=32, validation_data=(x_test,y_test))

        import numpy 
cvscores2 = []
scores = model2.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
cvscores2.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores2), numpy.std(cvscores2)))


