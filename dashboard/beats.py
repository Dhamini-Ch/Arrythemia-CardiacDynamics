import this
from nbformat import write
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import percentile
from numpy.random import rand





test = pd.read_csv("mitbih_test.csv", header=None)
train = pd.read_csv("mitbih_test.csv", header=None)
test.head()

#Finding the data size
print(train.shape)





#Getting the summary of the data
train.info()
#Displaying the unique classes in the dataset
train[187].unique()
#Finding the null values
count0 = train[0].isna().sum()
print("Number of null values for class 0: ",count0)
count1 = train[1].isna().sum()
print("Number of null values for class 1: ",count1)
count2 = train[2].isna().sum()
print("Number of null values for class 2: ",count2)
count3 = train[3].isna().sum()
print("Number of null values for class 3: ",count3)
count4 = train[4].isna().sum()
print("Number of null values for class 4: ",count4)

#Summarizing data
train.describe()

#Symmetry of the data
train.skew()

#Fining the outliers
continous_features =   train[187].unique()
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
            train.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))
outliers(train[continous_features])

#Removing the outliers
outliers(train[continous_features], drop=True)


#Finding the number of classes present in the data 
label_names = ['Non-ecotic beats', 'Supraventricular ectopic beats', 'Ventricular ectopic beats', 'Fusion beats', 'Unknown beats']
labels = train[187]
print("Count in each label: ")
print(labels.value_counts())
color1 = ['red', 'yellow', 'blue', 'green', 'black']

type_of_beats=plt.figure(figsize=(13,4))
plt.bar(list(set(label_names)), list(labels.value_counts()), color=color1, align='center')
plt.title('Beats')
plt.xlabel('Types of Beats')
plt.ylabel('Total number of collactions')




train.skew()

sns.set_style('whitegrid')
non_ect_beats=plt.figure(figsize=(12,4))
plt.plot(train.iloc[0, 0:187], color='red', label="Non-Ectopic Beats")
plt.xlabel("Time(ms)")
plt.ylabel("Amplitude")
plt.title("Non-Ecotic Beats")

#ECG of Supraventricular ectopic Beats
sns.set_style('whitegrid')
spr_ven_beats =plt.figure(figsize=(12,4))
plt.plot(train.iloc[1, 0:187], color='green', label="Supraventricular ectopic Beats")
plt.xlabel("Time(ms)")
plt.ylabel("Amplitude")
plt.title("Supraventricular ectopic beats")


#ECG of Ventricular ectopic beats
sns.set_style('whitegrid')
ven_ect_beats=plt.figure(figsize=(12,4))
plt.plot(train.iloc[1, 0:187], color='blue', label="Ventricular ectopic beats")
plt.xlabel("Time(ms)")
plt.ylabel("Amplitude")
plt.title("Ventricular ectopic beats")



#ECG of Fusion Beats 
sns.set_style('whitegrid')
fus_beats=plt.figure(figsize=(12,4))
plt.plot(train.iloc[1, 0:187], color='black', label="Fusion beats")
plt.xlabel("Time(ms)")
plt.ylabel("Amplitude")
plt.title("Fusion beats")



#ECG of unknown beats present in the data
sns.set_style('whitegrid')
unk_beats=plt.figure(figsize=(12,4))
plt.plot(train.iloc[1, 0:187], color='violet', label="Unknown beats")
plt.xlabel("Time(ms)")
plt.ylabel("Amplitude")
plt.title("Unknown beats")



# st.header("Classification of Arrhythmia")
# beats=st.sidebar.selectbox(label=("patient id"),options=(
#     "Non-ecotic beats",
#     "Ventricular ectopic beats",
#     "Supraventicular ectopic beats",
#     "Fusion beats",
#     "Unknown beats")
# )

def display_beats():
    st.write("The types of Arrhthymia")
    st.write("1.Ventricular Arrhythmias - Fusion Beats")
    st.write("2.Ectopic Rhythm(atrial fibrillation) - Ectopic Beats ")
    st.write("3.TachycardiaExtra Arrhythmias - Ventricular Ectopic Beats")
    st.write("4.Type-Tachycardia - Supraventricular Beats ")
    st.header("Dynamics of heartbeat")
    st.pyplot(type_of_beats)
    st.header("Unkown Beats")
    st.pyplot(unk_beats)
    st.write("These beats are unrecognised abnormal beats ")
    st.header("Ventricular ectopic beats - TachycardiaExtra ")
    st.pyplot(ven_ect_beats)
    st.write("ventricular ectopic: tachycardia"
"Extra, abnormal heartbeats that begin in one of the heart's two lower chambers."
"Premature ventricular contractions (PVCs) occur in most people at some point. Causes may include certain medication, alcohol, some illegal drugs, caffeine, tobacco, exercise or anxiety."
"PVCs often cause no symptoms. When symptoms do occur, they feel like a flip-flop or skipped-beat sensation in the chest."
"Most people with isolated PVCs and an otherwise normal heart don't need treatment. PVCs occurring continuously for longer than 30 seconds is a potentially serious cardiac condition known as ventricular tachycardia.")
    st.header("Supraventicular ectopic beats - Type-Tachycardia")
    st.pyplot(spr_ven_beats)
    st.write("supraventricular:  type-tachycardia  A faster than normal heart rate beginning above the hearts two lower chambers. Supraventricular tachycardia is a rapid heartbeat that develops when the normal electrical impulses of the heart are disrupted. There may be symptoms like heart palpitations or there may be no symptoms at all. Certain manoeuvres, medication, an electrical shock to the heart (cardioversion) and catheter-based procedures (ablation) can help slow the heart.")
    st.header("Fusion Beats - Ventricular Arrhythmias")
    st.pyplot(fus_beats)
    st.write("A fusion beat occurs when a supraventricular and a ventricular impulse coincide to produce a hybrid complex")
    st.header("Ectopic Beats - Atrial fibrillation")
    st.pyplot(non_ect_beats)
    st.write(
"Ectopic heartbeats are extra heartbeats that occur just before a regular beat. Ectopic beats are normal and usually not a cause for concern, though they can make people feel anxious. Ectopic beats are common. People may feel like their heart is skipping a beat or is producing an extra beat.")
    



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


