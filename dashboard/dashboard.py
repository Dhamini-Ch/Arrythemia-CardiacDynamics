#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# In[5]:


dfs = [pd.read_csv(x + '.csv') for x in ['mitbih_train','NormalAndAbnormal']]


# In[6]:


st.title('CLASSIFYING HEARTBEAT USING ELECTROCARDIOGRAM')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('MITBIH', 'PTBDB')
)



st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)


# In[40]:


mitbih=pd.read_csv("mitbih_train.csv")
ptbdb=pd.read_csv("NormalAndAbnormal.csv")
ptbdb = ptbdb.rename({187: 'target'}, axis = 1)
def get_dataset(name):
    data = None
    if name == 'MITBIH':
        data = dfs[0]
    else:
        data = dfs[1]
    X = data.iloc[: , : -1]
    y = data.iloc[:,-1]
    return X, y


# In[41]:


X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))


# In[42]:

@st.cache
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

#accuracy_metrics = st.sidebar.selectbox(
#    'ROC Curves',
#    ('Confusion Matrix', 'Classification Report', 'ROC Curve')
#)
# In[43]:




params = add_parameter_ui(classifier_name)

@st.cache
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf


# In[44]:


clf = get_classifier(classifier_name, params)


# In[45]:


X.values.reshape(-1,1)


# In[46]:


y.values.reshape(-1,1)


# In[47]:


#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write('Confusion Matrix: ')
st.write(confusion_matrix(y_test, y_pred))
st.write('Classification Report')
report=classification_report(y_test, y_pred, output_dict=True)
FinalReport=pd.DataFrame(report).transpose()
st.write(FinalReport)

# In[48]:


#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

#plt.show()
st.pyplot(fig)


# In[ ]:




