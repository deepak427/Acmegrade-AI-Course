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
X_test = X_test.astype('float32')
X_train/=255.0
X_test/=255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

# Creating the model

model=Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout (0.25))

model.add(Conv2D (64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout (0.25))

model.add(Flatten())
model.add(Dense (512, activation= 'relu'))
model.add(Dropout (0.5))
model.add(Dense (10, activation='softmax'))

# Compile the model

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print(model.summary())

# Train the model

model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(X_test,y_test),
    shuffle=True
)

# Save the neural network architecture
model_structure=model.to_json()
f=Path("model_structure.json")
f.write_text(model_structure)

# save the model

model.save_weights("May/data/model_weights.h5")

# Making predictions on image

from tensorflow.keras.models import model_from_json
from keras.preprocessing import image

class_labels=[
    "Plans",
    "car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]

f=Path("model_structure.json")
model_structure=f.read_text()

model=model_from_json(model_structure)

# Load an image file to test

import matplotlib.pyplot as plt
from tensorflow.keras.utlis import load_img, img_to_array

img =load_img("dog.png", target_size = (32,32))
plt.imshow(img)
plt.show()

# Convert the image to numpy array

import numpy as np

image_to_test = img_to_array(img)
list_of_images = np.expand_dims(image_to_test, axis=0)

results=model.predict(list_of_images)

single_result=results[0]

most_likely_class_index = int(np.argmax(single_result))
class_likelihood=single_result[most_likely_class_index]

class_label=class_labels[most_likely_class_index]

print("class label: {}".format(class_label))
print("class likelihood: {}".format(class_likelihood))