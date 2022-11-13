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

train = pd.read_csv("combined_train.csv", header=None)
test = pd.read_csv("combined_test.csv", header=None)

train
train.shape
train.describe()

label_names = ['Non-ecotic beats (normal beat)', 'Supraventricular ectopic beats', 'Ventricular ectopic beats', 'Fusion beats', 'Unknown beats', 'Abnormal', 'Normal']

labels = train[187].astype('int64') 

print("Count in each label: ")
print(labels.value_counts())

plt.barh(list(set(labels)), list(labels.value_counts()))

train.shape
train_lbl0 = resample(train[train[187]==0], replace=True, n_samples=50000, random_state=113)
train_lbl1 = resample(train[train[187]==1], replace=True, n_samples=50000, random_state=113)
train_lbl2 = resample(train[train[187]==2], replace=True, n_samples=50000, random_state=113)
train_lbl3 = resample(train[train[187]==3], replace=True, n_samples=50000, random_state=113)
train_lbl4 = resample(train[train[187]==4], replace=True, n_samples=50000, random_state=113)
train_lbl5 = resample(train[train[187]==5], replace=True, n_samples=50000, random_state=113)
train_lbl6 = resample(train[train[187]==6], replace=True, n_samples=50000, random_state=113)

train= pd.concat([train_lbl0, train_lbl1, train_lbl2, train_lbl3, train_lbl4, train_lbl5, train_lbl6])

labels = train[187].astype('int64')   # last column has the labels

print("Count in each label: ")
print(labels.value_counts())

plt.plot(np.array(train_lbl0.sample(1))[0, :187])
plt.title(label_names[0])

plt.plot(np.array(train_lbl1.sample(1))[0, :187])
plt.title(label_names[1])

plt.plot(np.array(train_lbl2.sample(1))[0, :187])
plt.title(label_names[2])

plt.plot(np.array(train_lbl3.sample(1))[0, :187])
plt.title(label_names[3])

plt.plot(np.array(train_lbl4.sample(1))[0, :187])
plt.title(label_names[4])

plt.plot(np.array(train_lbl5.sample(1))[0, :187])
plt.title(label_names[5])

plt.plot(np.array(train_lbl6.sample(1))[0, :187])
plt.title(label_names[6])


def gaussian_noise(signal):
    noise = np.random.normal(0,0.05,187)
    return signal + noise

sample = train_lbl0.sample(1).values[0]

sample_with_noise = gaussian_noise(sample[:187])

plt.subplot(1, 1, 1)

plt.plot(sample[:187])
plt.plot(sample_with_noise)


ytrain = tensorflow.keras.utils.to_categorical(train[187])
ytest = tensorflow.keras.utils.to_categorical(test[187])

# Input to the model
xtrain = train.values[:, :187]
xtest = test.values[:, :187]

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
model.add(Dense(7, activation = 'softmax'))

model.compile(optimizer = tensorflow.keras.optimizers.Adam(0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(xtrain, ytrain, epochs = 8, batch_size = 32, validation_data = (xtest, ytest))

def plot(history, variable, variable2):
    plt.plot(range(len(history[variable])), history[variable])
    plt.plot(range(len(history[variable2])), history[variable2])
    plt.legend([variable, variable2])
    plt.title(variable)

plot(history.history, "accuracy", "val_accuracy")
plot(history.history, "accuracy", "val_accuracy")

ypred = model.predict(xtest)

cm = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

for i in range(cm.shape[1]):
    for j in range(cm.shape[0]):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="black")


plt.imshow(cm, cmap=plt.cm.Blues)

print("The distribution of test set labels")
print(test[187].value_counts())

print('F1_score = ', f1_score(ytest.argmax(axis=1), ypred.argmax(axis=1), average = 'macro'))
# import csv
# f = open('path/to/csv_file', 'w')
# writer = csv.writer(f)
# writer.writerow(row)
# f.close()

#i = random.randint(0, len(xtest)-1)
for i in range(0, len(xtest)-1):
      output = model(np.expand_dims(xtest[i], i))

      pred = output.numpy()[i]

      plt.plot(xtest[i])

      print("Actual label: ", label_names[np.argmax(ytest[i])])
      print("Model prediction : ", label_names[np.argmax(pred)], " with probability ", pred[np.argmax(pred)])




