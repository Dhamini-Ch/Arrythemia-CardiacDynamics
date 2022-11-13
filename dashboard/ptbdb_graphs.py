import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import percentile
from numpy.random import rand

#Loading the ptbdb dats set
abnormal = pd.read_csv("D:/studies/2nd yr internship/New folder/dashboard/ptbdb_abnormal.csv", header = None)
normal = pd.read_csv("D:/studies/2nd yr internship/New folder/dashboard/ptbdb_normal.csv", header = None)

#Peaking at the data
abnormal.head()

#peaking at the data
normal.head()

#Displaying the unique classes in the dataset
abnormal[187].unique()

#Displaying the unique classes in the dataset
normal[187].unique()

# Displaying the dimensions of the data
print("Shape of Abnormal Data set",abnormal.shape)
print("Shape of normal Data set  ",normal.shape)

#Finding the null values
count0 = abnormal[0].isna().sum()
print("Number of null values for class 0(normal   : ",count0)
count1 = normal[1].isna().sum()
print("Number of null values for class 1(abnormal): ",count1)

#Finding the null values
normal.isnull().sum()

#Getting the summary of the data frame 
abnormal.info()

#Getting the summary of the data frame 
normal.info()

#Statistical Analysis
abnormal.describe()

#Summarizing data
normal.describe()

#Finding the symmetry of the data
abnormal.skew()

#Finding the symmetry of the data
normal.skew()

#Finding the data types of the columns
abnormal.dtypes

#Finding the data types of the columns
normal.dtypes

#Fining the outliers

continous_features =  abnormal[187].unique()
def outliers(df_out, drop = False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.) # 25th percentile of the data of the given feature
        Q3 = np.percentile(feature_data, 75.) # 75th percentile of the data of the given feature
        IQR = Q3-Q1 #Interquartile Range
        outlier_step = IQR * 1.5 #That's we were talking about above
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            abnormal.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))
outliers(abnormal[continous_features])


continous_features =  normal[187].unique()
def outliers(df_out, drop = False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.) # 25th percentile of the data of the given feature
        Q3 = np.percentile(feature_data, 75.) # 75th percentile of the data of the given feature
        IQR = Q3-Q1 #Interquartile Range
        outlier_step = IQR * 1.5 #That's we were talking about above
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            normal.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))
outliers(normal[continous_features])

#Removing the outliers
outliers(abnormal[continous_features], drop=True)

#Removing the outliers
outliers(normal[continous_features], drop=True)

#ECG of Abnormal beats 
sns.set_style('whitegrid')
abnormal =plt.figure(figsize=(12,4))
plt.plot(abnormal.iloc[1, 0:187], color = 'red', label='Abnormal Beats')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Abnormal Beats")
plt.show()

#ECG of normal beats 

sns.set_style('whitegrid')
normal = plt.figure(figsize=(12,4))
plt.plot(normal.iloc[0, 0:187], color = 'green', label='Normal Beats')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Normal Beats")
plt.show()

#Correlation of the data
plt.figure(figsize=(20,10))
c = abnormal.corr()
heat = sns.heatmap(c)
c

#Bar plot for abnormal data 
sns.set_style("whitegrid")
plt.subplot(2,3,1)
a0 = sns.boxplot(x=abnormal[1], data=abnormal)

#Bar plot for normal data
plt.subplot(2,3,2)
a1 = sns.boxplot(x=normal[0], data=normal)


#5-number summary for abnormal class
quartiles = percentile(abnormal[1], [25, 50, 75])
data_min, data_max = abnormal[1].min(), abnormal[1].max()
print('Minimum : ', data_min)
print('Q1      : ', quartiles[0])
print('Median  : ', quartiles[1])
print('Q3      : ', quartiles[2])  
print('Maximum : ', data_max)

#Barplot for abnormal class
a2 = sns.boxplot(x=abnormal[1], data=abnormal)

#5-number summary for normal class
quartiles = percentile(normal[0], [25, 50, 75])
data_min, data_max = normal[0].min(), normal[0].max()
print('Minimum : ', data_min)
print('Q1      : ', quartiles[0])
print('Median  : ', quartiles[1])
print('Q3      : ', quartiles[2])  
print('Maximum : ', data_max)

#Barplot for normal class 
a3 = sns.boxplot(x=normal[0], data=abnormal)

def ptbdb():
    st.pyplot(abnormal)

    st.pyplot(normal)

    st.pyplot(heat) 

    st.pyplot(a0)

    st.pyplot(a1)

    st.pyplot(a2)

    st.pyplot(a3)

ptbdb()