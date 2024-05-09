# Chatbot using NLP and neural  networks in python

from tensorflow.keras.models import Sequential
import string
import json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow. keras import Sequential
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
import random
data = {"intents": [
    {"tag": "greeting",
     "patterns": ["Hello", "How are you?", "Hi There", "Hi", "What's up"],
     "responses": ["Howdy Partner!", "Hello", "How are you doing?", "Greetings!", "How do you do"]
     },
    {"tag": "age",
     "patterns": ["how old are you", "when is your birthday", "when was you born"],
     "responses": ["I am 24 years old", "I was born in 1966", "My birthday is July 3rd and I was born in 1996", "03/07/1996"]},
    {"tag": "date",
     "patterns": ["what are you doing this weekend", "do you want to hangout sometime?", "what are your plans for this week"],
     "responses": ["I am available this week", "I don't have any plans", "I am not busy"]
     },
    {"tag": "name",
     "patterns": ["what's your name", "what are you called", "who are you"],
     "responses": ["My name is Kippi", "I'm Kippi", "Kippi"]
     },
    {"tag": "goodbye",
     "patterns": ["bye", "g2g", "see ya", "adios", "cya"],
     "responses": ["It was nice speaking to you", "See you later", "Speak Soon"]},
]}

# import dependencies


nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
words = []
classes = []
doc_x = []
doc_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Lemetization
words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in string.punctuation]

words = sorted(set(words))
classes = sorted(set(classes))

# List for training data

training = []
out_empty = [0]*len(classes)

# creating a bag of words model
for idx, doc in enumerate(doc_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    training.append([bow, output_row])

random.shuffle(training)

training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])

epochs = 500

# Creating model

# Create a Sequential model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation='softmax'))

# Create the Adam optimizer with a specified Learning rate
adam = tf.keras.optimizers. Adam(learning_rate=0.01)

# Compile the model using the Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())

model.fit(x=train_X, y=train_y, epochs=epochs, verbose=1)

def clean_text(text):
    tokens=nltk.word_tokenize(text)
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens=clean_text(text)
    bow=[0]*len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word==w:
                bow[idx]=1
    return np.array(bow)

def pred_class (text, vocab,labels): 
    bow=bag_of_words (text, vocab) 
    result=model.predict(np.array([bow]))[0]
    thresh=0.2
    y_pred=[[idx, res] for idx, res in enumerate (result) if res>thresh]

    y_pred.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json): 
    tag=intents_list[0]
    list_of_intents=intents_json["intents"] 
    for i in list_of_intents: 
        if i["tag"]==tag:
            result=random.choice (i["responses"])
            break
    return result

while True:
    message=input("")
    intents=pred_class(message.lower(), words, classes)
    result=get_response(intents, data)
    print(result)