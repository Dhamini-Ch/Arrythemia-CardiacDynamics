import streamlit as st
from table import table

# with open ("style.css") as f:
#     st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

from argparse import _HelpAction
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
 
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

from beats import *
from abnormal import *
import pickle
# from ada_boost_mitbih import ada_mitbih
# from ada_boost_mitptb import ada_mitbptb
# import ann_mitbih 
# import ann_ptbdb
# import ann_mitptb
# import cnn_mitbih
# import cnn_ptbdb
# import cnn_mitptb

# import ada_boost_ptbdb
# import ada_boost_mitptb
# from xgboost_mitbih import print_xgaccuracy
# import xgboost_ptbdb
# import xgboost_mitpdb




dfs = [pd.read_csv(x + '.csv') for x in ['mitbih_train','NormalAndAbnormal', 'combined']]



# In[6]:


st.title('Assessment of Cardiac Dynamics')




st.subheader("""
 Evaluation of Machine Learning models for classification of Heart Beat
""")

#classfication graphs's

st.sidebar.title('Visualization of Datasets')

st.sidebar.button("Visualization of MITBIH dataset", on_click=display_beats)

st.sidebar.button("Visualization of PTBDB dataset",on_click=ptbdb)
  


# classifiers option

st.sidebar.title('Different Classifiers')
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('MITBIH', 'PTBDB','MITBIH + PTBDB')
)



st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select Base classifier',
    ('KNN', 'SVM', 'Random Forest')
)

if classifier_name == 'Random Forest':
    boosting_name = st.sidebar.selectbox(
    'Select Boosting  ',
    ('XG Boost','ADA Boost')
)

neural_model = st.sidebar.selectbox(
    'Select Deep Learning Classifier',
    ('ANN (homogeneous)','CNN')
    
)


# In[40]:


mitbih=pd.read_csv("mitbih_train.csv")
ptbdb=pd.read_csv("NormalAndAbnormal.csv")

ptbdb = ptbdb.rename({187: 'target'}, axis = 1)
    
def get_dataset(name):
    data = None
    if name == 'MITBIH':
        data = dfs[0]
    elif name == 'PTBDB':
        data = dfs[1]
    else:
        data = dfs[2]    
    X = data.iloc[: , : -1]
    y = data.iloc[:,-1]
    return X, y



X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))



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







params = add_parameter_ui(classifier_name)

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

y.values.reshape(-1,1)


#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

# pickled_model = pickle.load(open('ANN_MIT-BIH', 'rb'))
# pickled_model.predict(x_test)


st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
col3 , col4 = st.columns(2)
col3.write('Confusion Matrix:')
col4.write('Classification Report:')
col1 , col2  = st.columns(2)
col1.write(confusion_matrix(y_test, y_pred))

report=classification_report(y_test, y_pred, output_dict=True)
FinalReport=pd.DataFrame(report).transpose()
col2.write(FinalReport)






# BOOSTING ACCURACY
def boosting_acc(boost_name,data_name):
    st.subheader('Boosting')
    if boost_name == "XG Boost":
        if data_name == "MITBIH":
            st.write("Accuracy after boosting: 93.54")
        elif data_name == "PTBDB":
            st.write("Accuracy after boosting: 99.32")
        else:
            st.write("Accuracy after boosting: 99.42")  
    elif boost_name == "ADA Boost":
        if data_name == "MITBIH":
            st.write("Accuracy after boosting: 58.57")
        elif data_name == "PTBDB":
            st.write("Accuracy after boosting: 87.32")
        else:
            st.write("Accuracy after boosting: 47.15")

if classifier_name == 'Random Forest':
    boosting_acc(boosting_name,dataset_name)


#deep learning accuracy
def deep_learning_accuracy(model_name,dataset_name):
    st.subheader("Accuracy from the deep learning model")
    if model_name == "ANN (homogeneous)":
        if dataset_name == 'MITBIH':
            st.write("Accuracy from Artificial Neural Networks: 84.81 ")
        elif dataset_name == 'PTBDB':
            st.write ('Accuracy from Artificial Neural Networks: 94.78')
        else:
            st.write ('Accuracy from Artificial Neural Networks: 84.24')
    elif model_name == 'CNN':
        if dataset_name == 'MITBIH':
            st.write("Accuracy from Convolutioonal Neural Networks: 98.78 ")
        elif dataset_name == 'PTBDB':
            st.write ('Accuracy from Convolutional Neural Networks: 98.43')
        else:
            st.write ('Accuracy from Convolutional Neural Networks: 99.04')

deep_learning_accuracy(neural_model,dataset_name)

st.sidebar.button('Accuracy for all the models',on_click=table)
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

st.subheader("Heat Map for the dataset")
#plt.show()
st.pyplot(fig)

# copyrights

st.markdown("ANALYSIS OF CARDIAC DYNAMICS AND ASSESSMENT OF ARRHYTHMIA BY CLASSFYING HEARTBEAT USING ELECTROCARDIOGRAM  ")

# hide_streamlit_style = """
# <style>

# footer {visibility: hidden;}
# footer:after{content:'';}

                
                
# </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

copyright_="""
<hr>
Copyrights <br>
All Rights Reserved
<hr>
Project By:
Medha,Dhamin,Mukesh,Rishikesh
"""
st.markdown(copyright_,unsafe_allow_html=True)


