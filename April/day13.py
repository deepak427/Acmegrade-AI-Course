# NLP: Ability of computer to understand, analyze, manupulate and potentialy generate human language

# import dependencies

import pandas as pd
import nltk
nltk.download('stopwords')
import numpy as np

dataset=pd.read_csv("April/data/spam.csv", header=None, encoding="latin-1")
dataset.drop(dataset.columns[[2, 3, 4]], axis=1, inplace=True)
dataset.columns=['label', 'body_text']
print(dataset.head())

# shape of data

print("Input data has {} rows and {} columns".format(len(dataset), len(dataset.columns)))

print("Out of {} rows, {} are spam, {} are ham".format(len(dataset), len(dataset[dataset["label"]=="spam"]),
                                                       len(dataset[dataset["label"]=="ham"])))

# Preprocessing

print("Number of null in label: {}".format((dataset["label"].isnull().sum())))
print("Number of null in text: {}".format((dataset["body_text"].isnull().sum())))

import string

# Removing punctuations and tokenizing data

def remove_punc(text):
    text_nopunc = "".join([char for char in text if char not in string.punctuation])
    return text_nopunc

dataset["body_text_clean"] = dataset["body_text"].apply(lambda x: remove_punc(x))
print(dataset.head())

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

# Stemming

ps=nltk.PorterStemmer()

def stemming(tokenized_text):
    text=[ps.stem(word) for word in tokenized_text]
    return text

dataset["body_text_stemmed"]=dataset["body_text_nostop"].apply(lambda x:stemming(x))
print(dataset.head())

wn=nltk.WordNetLemmatizer()

def lematizing(tokenized_text):
    text=[wn.lemmatize(word) for word in tokenized_text]
    return text

dataset["body_text_lematiazed"]=dataset["body_text_nostop"].apply(lambda x:lematizing(x))
print(dataset.head())

# Vectorization

from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    tokens=re.split("\W", text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text

count_vect = CountVectorizer(analyzer=clean_text)
X_count=count_vect.fit_transform(dataset['body_text'])
print(X_count.shape)
# Apply count vec to sample data

data_sample=dataset[0:20]
count_vect_sample =CountVectorizer(analyzer=clean_text)
X_count_sample= count_vect_sample.fit_transform(data_sample['body_text'])

X_counts_df = pd.DataFrame(X_count_sample.toarray())

import warnings
warnings.filterwarnings("ignore")

X_counts_df.columns=count_vect_sample.get_feature_names_out()
print(X_counts_df)
# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(dataset["body_text"])

print(X_tfidf.shape)

# testing tfidf on small sample

data_sample=dataset[0:20]

tfidf_vect_sample =TfidfVectorizer(analyzer=clean_text)
X_tfidf_sample= tfidf_vect_sample.fit_transform(data_sample['body_text'])

X_tfidf_df = pd.DataFrame(X_tfidf_sample.toarray())
X_tfidf_df.columns=tfidf_vect_sample.get_feature_names()
print(X_tfidf_df)

# Feature engineering

dataset=pd.read_csv("April/data/spam.csv", header=None, encoding="latin-1")
dataset.drop(dataset.columns[[2, 3, 4]], axis=1, inplace=True)
dataset.columns=['label', 'body_text']

dataset["body_len"]=dataset["body_text"].apply(lambda x:len(x)-x.count(" "))

def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3) * 100

dataset["punct%"]=dataset["body_text"].apply(lambda x:count_punct(x))
print(dataset.head())

import matplotlib.pyplot as plt
import numpy as np

bins=np.linspace(0,200,40)

plt.hist(dataset['body_len'], bins)
plt.title('Body length distribution')
plt.show()

bins=np.linspace(0,50,40)

plt.hist(dataset['punct%'], bins)
plt.title('punc perc. distribution')
plt.show()
# Building machine learning classifier

# model using k-fold cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(dataset["body_text"])

X_features=pd.concat([dataset['body_len'], dataset['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
print(X_features)

rf=RandomForestClassifier(n_jobs=1)
k_fold=KFold(n_splits=5)

print(cross_val_score(rf, X_features, dataset["label"], cv=k_fold, scoring="accuracy", n_jobs=1))

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_features, dataset["label"], test_size=0.10)

# Creating random forest classifer using train test split

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=500, max_depth= 20, n_jobs=1)
rf_model=rf.fit(X_train, Y_train)

sorted(zip(rf_model.feature_importances_, X_train.columns,), reverse=True)[0:10]

y_pred=rf_model.predict(X_test)

Precision, recall, fscore, support = score(Y_test, y_pred, pos_label='spam', average='binary')

print("Precision {}/ Recall {}/ Accuracy {}".format(round(Precision,3),
                                                    round(recall,3),
                                                    round((y_pred==Y_test).sum()/len(y_pred),3)))
