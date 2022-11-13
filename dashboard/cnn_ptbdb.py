import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample
import tensorflow
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, MaxPool1D, Convolution1D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random

#Loading the ptbdb dats set
abnormal     = pd.read_csv("ptbdb_abnormal.csv", header = None)
normal       = pd.read_csv("ptbdb_normal.csv", header = None)

abnormal.shape

normal.shape

abnormal[187].unique()

normal[187].unique()

#Combining abnormal and normal
ptbdb_data = pd.merge(abnormal, normal, how='outer')
ptbdb_data

ptbdb_data_lbl0 = resample(ptbdb_data[ptbdb_data[187]==0], replace=True, n_samples=15000, random_state=113)
ptbdb_data_lbl1 = resample(ptbdb_data[ptbdb_data[187]==1], replace=True, n_samples=15000, random_state=113)

label_names = ['Normal Beats', 'Abnormal Beats']
train= pd.concat([ptbdb_data_lbl0, ptbdb_data_lbl1])
labels = train[187].astype('int64')   # last column has the labels

print("Count in each label: ")
print(labels.value_counts())

ptbdb_data.shape

#Feature Scaling
x = ptbdb_data.iloc[:, :-1].values
y = ptbdb_data.iloc[:, -1].values

x.shape

#Splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

plt.plot(np.array(ptbdb_data_lbl0.sample(1))[0, :187])
plt.title(label_names[0])

plt.plot(np.array(ptbdb_data_lbl1.sample(1))[0, :187])
plt.title(label_names[0])

def gaussian_noise(signal):
    noise = np.random.normal(0,0.05,187)
    return signal + noise

sample = ptbdb_data_lbl1.sample(1).values[0]

sample_with_noise = gaussian_noise(sample[:187])

plt.subplot(1, 1, 1)

plt.plot(sample[:187])
plt.plot(sample_with_noise)


ytrain = tensorflow.keras.utils.to_categorical(y_train[187])
ytest = tensorflow.keras.utils.to_categorical(y_test[187])

# Input to the model
xtrain = x_train
xtest = x_test

# Adding noise
for i in range(xtrain.shape[0]):
    xtrain[i, :187] = gaussian_noise(xtrain[i, :187])

xtrain.shape
xtrain

xtrain = np.expand_dims(xtrain, 2)
xtest = np.expand_dims(xtest, 2)

print("Shape of training data: ")
print("Input: ", xtrain.shape)
print("Output: ", ytrain.shape)

print("\nShape of test data: ")
print("Input: ", xtest.shape)
print("Output: ", ytest.shape)

model = Sequential()
model.add(Conv1D(64, 6, activation = 'relu', input_shape = xtrain[0].shape))
model.add(MaxPool1D(3, 2))

model.add(Conv1D(64, 6, activation = 'relu'))
model.add(MaxPool1D(3, 2))

model.add(Conv1D(64, 6, activation = 'relu'))
model.add(MaxPool1D(3, 2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'softmax'))

model.compile(optimizer = tensorflow.keras.optimizers.Adam(0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


history = model.fit(x_train, y_train, epochs = 8, batch_size = 32)

def plot(history, variable, variable2):
    plt.plot(range(len(history[variable])), history[variable])
    plt.plot(range(len(history[variable2])), history[variable2])
    plt.legend([variable, variable2])
    plt.title(variable)
    

