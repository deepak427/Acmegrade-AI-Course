# Deep learning framework: Number of weights = no of feautures

# ANN: Each node has its own weights, biases, and activation function

# Typical node count for hidden layer: 2^n

# Import dependencies

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

iris_data = sns.load_dataset("iris")
print(iris_data.head())

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
iris_data['species'] = label_encoder.fit_transform(iris_data["species"])
print(iris_data.head())

np_iris = iris_data.to_numpy()

X_data=np_iris[:,0:4]
Y_data=np_iris[:,4]

print("\n Features before scaling: \n------------->")
print(X_data[:5,:])

print("\n targets before one hot encoding: \n------------->")
print(Y_data[:5])

scalar = StandardScaler().fit(X_data)

X_data = scalar.transform(X_data)

Y_data = tf.keras.utils.to_categorical(Y_data, 3)
print(Y_data)

print("\n Features after scaling: \n------------->")
print(X_data[:5,:])

print("\n targets after one hot encoding: \n------------->")
print(Y_data[:5])

# Spiliting data

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.10)

print("\n Train test Dimensions: \n------>")
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Creating deep learning model

from tensorflow import keras

NB_classes = 3

model = tf.keras.models.Sequential()

# Input layer
model.add(keras.layers.Dense(128,
                             input_shape=(4,),
                             name="Input-Layer",
                             activation="relu"))

# second hidden layer
model.add(keras.layers.Dense(128,
                             name="Hidden-Layer-2",
                             activation="relu"))

# output layer
model.add(keras.layers.Dense(NB_classes,
                             name="Output-Layer",
                             activation="softmax"))

model.compile(loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

VERBOSE=1

BATCH_SIZE=16

EPOCHS=20

VALIDATION_SPLIT=0.2

print("\n Training Progress: \n-----------")

history=model.fit(X_train,
                  Y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=VERBOSE,
                  validation_split=VALIDATION_SPLIT)

print("\n Accuracy during Training: \n----------")

pd.DataFrame(history.history)['accuracy'].plot(figsize=(8,5))
plt.title("Accuracy improvement after each epoch")
plt.show()

print("\nEvaluate against test dataset: \n--------------")
print(model.evaluate(X_test, Y_test))

# Saving a model
# model.save("April/models/iris_save")

# # Load the model
# loaded_model = keras.models.load_model("iris_save")

# # display the model summary

# print(loaded_model.summary())

# Predictions with deep learning model

prediction_input = [[2.6,12.,2.4,4.4]]

# Scale the prediction datat with the same scaling object
scaled_input = scalar.transform(prediction_input)

# Get the raw prediction probabilities
raw_prediction = model.predict(scaled_input)
print("Raw prediction output (Probabilities): ", raw_prediction)

# Find predictions
prediction = np.argmax(raw_prediction)
print(prediction)
print("Prediction is ", label_encoder.inverse_transform([prediction]))


