# Chatbot using NLP and neural  networks in python

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

import json 
import string 
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

from tensorflow. keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download ("punkt") 
nltk.download ("wordnet")

lemmatizer=WordNetLemmatizer()
words=[] 
classes=[]
doc_x=[] 
doc_y=[]

for intent in data["intents"]: 
    for pattern in intent["patterns"]:
        tokens=nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"]) 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Lemetization
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

words=sorted(set(words))
classes=sorted(set(classes))

# List for training data

training=[]
out_empty=[0]*len(classes)

#creating a bag of words model
for idx, doc in enumerate(doc_x):
    bow=[]
    text=lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0) 
    output_row=list(out_empty)
    output_row[classes.index(doc_y[idx])]=1

    training.append([bow, output_row])

random.shuffle(training)

training=np.array(training, dtype=object)

train_X=np.array(list (training[:,0]))
train_y=np.array(list (training[:,1]))

