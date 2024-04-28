from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt

# Load the data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0], cmap="gray")
plt.show()
print(y_train[0])

# Preprocess the data

image_height, image_width = 28,28

print(X_train.shape)

X_train=X_train.reshape(60000, image_height*image_width)
X_test=X_test.reshape(10000, image_height*image_width)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

# normalize data

X_train/=255.0
X_test/=255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Building the model

model = Sequential()

model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model

model.compile(loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

plt.plot(history.history["accuracy"])
plt.show()

score=model.evaluate(X_test, y_test)
print(score)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(60000,28,28,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255.0
X_test/=255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN model developement



