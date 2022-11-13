#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import percentile
from numpy.random import rand



# In[2]:


#Loading the ptbdb dats set
abnormal = pd.read_csv("ptbdb_abnormal.csv", header=None)
normal = pd.read_csv("ptbdb_normal.csv", header=None)


# In[3]:


#Peaking at the data
abnormal.head()


# In[4]:


#peaking at the data
normal.head()


# In[5]:


#Displaying the unique classes in the dataset
abnormal[187].unique()


# In[6]:


#Displaying the unique classes in the dataset
normal[187].unique()


# In[7]:


# Displaying the dimensions of the data
print("Shape of Abnormal Data set",abnormal.shape)
print("Shape of normal Data set  ",normal.shape)


# In[8]:


#Finding the null values
count0 = abnormal[0].isna().sum()
print("Number of null values for class 0(normal   : ",count0)
count1 = normal[1].isna().sum()
print("Number of null values for class 1(abnormal): ",count1)


# In[9]:


#Getting the summary of the data frame 
abnormal.info()


# In[10]:


#Getting the summary of the data frame 
normal.info()


# In[11]:


#Statistical Analysis
abnormal.describe()


# In[12]:


#Summarizing data
normal.describe()


# In[13]:


#Finding the symmetry of the data
abnormal.skew()


# In[14]:


#Finding the symmetry of the data
normal.skew()


# In[15]:





# In[17]:


#Finding the outliers
continous_features =   abnormal[187].unique()
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
            abnormal.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))
outliers(abnormal[continous_features])


# In[18]:


#Removing the outliers
outliers(normal[continous_features], drop=True)


# In[19]:


#ECG of Abnormal beats 
sns.set_style('whitegrid')
abnor_beats=plt.figure(figsize=(12,4))
plt.plot(abnormal.iloc[1, 0:187], color = 'red', label='Abnormal Beats')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Abnormal Beats")
plt.show()



# In[20]:


#ECG of normal beats 

sns.set_style('whitegrid')
nor_beats=plt.figure(figsize=(12,4))
plt.plot(normal.iloc[0, 0:187], color = 'green', label='Normal Beats')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Normal Beats")
plt.show()



# In[21]:


#Correlation of the data
heat = plt.figure(figsize=(20,10))
c = abnormal.corr()
sns.heatmap(c)



# # In[22]:


#Box plot for abnormal data 
sns.set_style("whitegrid")
plt.subplot(2,3,1)
a0 = sns.boxplot(x=abnormal[1], data=abnormal)

#Box plot for normal data
plt.subplot(2,3,2)
a1 = sns.boxplot(x=normal[0], data=normal)


# # In[23]:


#5-number summary for abnormal class
quartiles = percentile(abnormal[1], [25, 50, 75])
data_min, data_max = abnormal[1].min(), abnormal[1].max()
print('Minimum : ', data_min)
print('Q1      : ', quartiles[0])
print('Median  : ', quartiles[1])
print('Q3      : ', quartiles[2])  
print('Maximum : ', data_max)

#Boxplot for abnormal class
a2 = sns.boxplot(x=abnormal[1], data=abnormal)


# # In[24]:


#5-number summary for normal class
quartiles = percentile(normal[0], [25, 50, 75])
data_min, data_max = normal[0].min(), normal[0].max()
print('Minimum : ', data_min)
print('Q1      : ', quartiles[0])
print('Median  : ', quartiles[1])
print('Q3      : ', quartiles[2])  
print('Maximum : ', data_max)

#Boxplot for normal class 
a3 = sns.boxplot(x=normal[0], data=abnormal)


def ptbdb():
    st.title("Visualization of PTBDB dataset")
    st.header("ECG of Abnormal beats")
    st.pyplot(abnor_beats)
    st.header("ECG of normal beats")
    st.pyplot(nor_beats)
    st.header("Heat Map for Correlation of the Data")
    st.write(heat)
    

    
    
 