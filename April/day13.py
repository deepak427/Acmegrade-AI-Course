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

def 



