import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRFClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import std
from numpy import mean

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

import streamlit as st



train=pd.read_csv("mitbih_train.csv", header=None)
test=pd.read_csv("mitbih_test.csv", header=None)
abnormal=pd.read_csv("ptbdb_abnormal.csv", header=None)
normal=pd.read_csv("ptbdb_normal.csv", header=None)

abnormal[187]=5
normal[187]=6

merged_data=pd.concat([train, test, abnormal, normal])

#Finding the outliers
continous_features =   merged_data[187].unique()
def outliers(df_out, drop = False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.) # 25th percentile of the data of the given feature
        Q3 = np.percentile(feature_data, 75.) # 75th percentile of the data of the given feature
        IQR = Q3-Q1 
        outlier_step = IQR * 1.5 
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            merged_data.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))
outliers(merged_data[continous_features])

#Removing the outliers
outliers(merged_data[continous_features], drop=True)

merged_data_lbl0 = resample(merged_data[merged_data[187]==0], replace=True, n_samples=15000, random_state=113)
merged_data_lbl1 = resample(merged_data[merged_data[187]==1], replace=True, n_samples=15000, random_state=113)
merged_data_lbl2 = resample(merged_data[merged_data[187]==2], replace=True, n_samples=15000, random_state=113)
merged_data_lbl3 = resample(merged_data[merged_data[187]==3], replace=True, n_samples=15000, random_state=113)
merged_data_lbl4 = resample(merged_data[merged_data[187]==4], replace=True, n_samples=15000, random_state=113)

merged_data= pd.concat([merged_data_lbl0, merged_data_lbl1, merged_data_lbl2, merged_data_lbl3, merged_data_lbl4])

labels = merged_data[187].astype('int64')   # last column has the labels

print("Count in each label: ")
print(labels.value_counts())

x=merged_data.iloc[:, 1:-1].values
y=merged_data.iloc[:, -1].values
# split into train test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

param = {
'max_depth': 5, # the maximum depth of each tree
'eta': 0.3, # the training step for each iteration
'silent': 1, # logging mode - quiet
'objective': 'multi:softprob', # error evaluation for multiclass training
'num_class': 7} # the number of classes that exist in this datset
num_round = 200 # the number of training iterations

bst = xgb.train(param, dtrain, num_round)

# make prediction
preds = bst.predict(dtest)
preds_rounded = np.argmax(preds, axis=1)
print(accuracy_score(y_test, preds_rounded))

# creative curve

# fit model on training data
model = XGBClassifier()
eval_set = [(x_train, y_train), (x_test, y_test)]
model.fit(x_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set,verbose=True)

# make predictions for test data
predictions = model.predict(x_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)

# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
st.pyplot(fig)

# plot classification error
fig1, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
st.pyplot(fig1)
st.markdown(accuracy)
