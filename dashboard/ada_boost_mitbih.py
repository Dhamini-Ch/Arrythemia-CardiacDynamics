#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
#Loading the ptbdb dats set
test  = pd.read_csv("mitbih_test.csv", header=None)
train = pd.read_csv("mitbih_train.csv", header=None)

test.shape

train.shape

#Finding the number of classes in the datset
train[187].unique()

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
import seaborn as sns
train_lbl0 = resample(train[train[187]==0], replace=True, n_samples=15000, random_state=113)
train_lbl1 = resample(train[train[187]==1], replace=True, n_samples=15000, random_state=113)
train_lbl2 = resample(train[train[187]==2], replace=True, n_samples=15000, random_state=113)
train_lbl3 = resample(train[train[187]==3], replace=True, n_samples=15000, random_state=113)
train_lbl4 = resample(train[train[187]==4], replace=True, n_samples=15000, random_state=113)


train= pd.concat([train_lbl0, train_lbl1, train_lbl2, train_lbl3, train_lbl4])
labels = train[187].astype('int64')   # last column has the labels

print("Count in each label: ")
print(labels.value_counts())

#Feature Scaling
x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

#Feature Scaling
x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

#Dimminsion of the dataset
x_test.shape

#Diminsion of the dataset
y_test.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


max_estimators = 50
ada_boost = AdaBoostClassifier(RandomForestClassifier(max_depth = 1, # Just a stump.
                                      random_state = np.random.RandomState(0)),
                               n_estimators = max_estimators,
                               random_state = np.random.RandomState(0))

# Fit all estimators.
ada_boost.fit(x_train, y_train)


mit_score = ada_boost.score(x_test, y_test)

st.markdown(mit_score)

def ada_mitbih():
       acc = st.write('accuracy after boosting:',mit_score)
       return acc

