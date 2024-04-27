# NLP: Ability of computer to understand, analyze, manupulate and potentialy generate human language

# import dependencies

import pandas as pd
import nltk
import numpy as np

dataset=pd.read_csv("April/data/spam.csv",sep='\t', header=None)
dataset.columns=['label', 'body_text']
print(dataset.head())

# shape of data

print("Input data has {} rows and {} columns".format(len(dataset), len(dataset.columns)))

print("Out of {} rows, {} are spam, {} are ham".format(len(dataset), len(dataset[dataset["label"]=="spam"]),
                                                       len(dataset[dataset["label"]=="ham"])))

# Preprocessing

print("Number of null in label: {}".format((dataset["label"].isnull().sum())))
print("Number of null in text: {}".format((dataset["text"].isnull().sum())))

import string

# Removing punctuations and tokenizing data

def remove_punc(text):
    text_nopunc = "".join([char for char in text if char not in string.punctuation])
    return text_nopunc

dataset["body_text_clean"] = dataset["text"].apply(lambda x: remove_punc(x))

import re

def tokenize(text):
    tokens=re.split('\W', text)
    return tokens

dataset["body_text_tokenized"]= dataset["body_text_clean"].apply(lambda x: tokenize(x.lower()))

print(dataset.head())

stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenizaed_list):
    text=[word for word in tokenizaed_list if word not in stopwords]
    return text

dataset["body_text_nostop"]= dataset["body_text_tokenized"].apply(lambda x:remove_stopwords(x))
print(dataset.head())


