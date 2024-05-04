# Image recognition using CNN on CIFAR Dataset

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
from tensorflow.keras.utils import to_categorical

# Load the data

(X_train, y_train),(X_test, y_test) = cifar10.load_data()

# Normalize the data

X_train = X_train.astype('float32')
X_test = X_train.astype('float32')
X_train/=255.0
X_test/=255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)