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

# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# iris_data['Species'] = label_encoder.fit_transform(iris_data["Species"])
# print(iris_data.head())

# np_iris = iris_data.to_numpy()

# scalar = StandardScaler().fit(X_data)

# X_data = scalar.transform(X_data)

# Y_data = keras.utils.to_categorical(Y_data, 3)

# print("\n Features after scaling: \n------------->")
# print(X_data[:5,:])

# print("\N T")
# print(Y_data[:5])

# # Spiliting data

# X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.10)

# print("\n Train test Dimensions: \n------>")
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# # Creating deep learning model

# from tensorflow import keras

# NB_classes = 3

# model = tf.keras.models.Sequential()

# model.add(keras)

# # Saving a model
# model.save("iris_save")

# # Load the model
# loaded_model = keras.models.load_model("iris_save")

# # display the model summary

# print(loaded_model.summary())

# # Predictions with deep learning model

# prediction_input = [[2.6,12.,2.4,4.4]]

# # Scale the prediction datat with the same scaling object
# scaled_input = scalar.transform(prediction_input)

# # Get the raw prediction probabilities
# raw_prediction = loaded_model.predict(scaled_input)
# print("Raw prediction output (Probabilities): ", raw_prediction)

# # Find predictions
# prediction = np.argmax(raw_prediction)
# print("Prediction is ", label_encoder.inverse_transform([prediction]))
