#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#Loading the ptbdb dats set
abnormal     = pd.read_csv("ptbdb_abnormal.csv", header = None)
normal       = pd.read_csv("ptbdb_normal.csv", header = None)

#Dimisions of abnormal dataset
abnormal.shape

#Diminsions of the normal dataset
normal.shape

#Classes in abnormal dataset
abnormal[187].unique()

#Clasess in the normal dataset
normal[187].unique()

#Combining abnormal and normal
ptbdb_data = pd.merge(abnormal, normal, how='outer')
ptbdb_data

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


max_estimators = 50
ada_boost = AdaBoostClassifier(RandomForestClassifier(max_depth = 1, # Just a stump.
                                      random_state = np.random.RandomState(0)),
                               n_estimators = max_estimators,
                               random_state = np.random.RandomState(0))

# Fit all estimators.
ada_boost.fit(x_train, y_train)

ptb_score=ada_boost.score(x_test, y_test)
st.markdown(ptb_score)
