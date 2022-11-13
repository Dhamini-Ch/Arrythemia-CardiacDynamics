import streamlit
import streamlit as st
import pandas as pd
 
# initialize data of lists.
data = {'MITBIH':[84.39,83.98,90.81,90.64,93.54,58.57,84.81,98.78],
        'PTBDB':[91.14,82.71,95.95,93.84,99.32,87.32,94.78,98.43],
        'MITBIH + PTBDB':[84.31,90.63,95.27,90.87,99.42,47.15,84.28,99.04]}

df = pd.DataFrame(data,index=['KNN','SVM','RANDOM FOREST','VOTING CLASSIFIER','XGBOOST','ADA BOOST',
'ANN','CNN'])    

def table():
    st.title('Accuracy of all the models')
    st.table(df)
    return df