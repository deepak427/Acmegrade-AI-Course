# Data Preprocessing
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

## Feature Scaling

### Standarization

from sklearn.preprocessing import StandardScaler

data = sns.load_dataset("mpg")

scalar=StandardScaler()
# scalar.fit(data)

# data_scaled=scalar.transform(data)

# print(data.head())

# sns.kdeplot(data=["age"])

# Min max scalar

from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()
# scalar.fit(data)

# data_scaled=scalar.transform(data)
# print(data.head())

# Handling Missing Values
# data.isnull.mean

data=data[["mpg","cylinders","displacement"]]
print(data.isnull().mean())

# median=data.mpg.meadian()
# print(median)

# Frequent Category Imputation

titanic_data= sns.load_dataset("titanic")

titanic_data= titanic_data[["embark_town","age","fare"]]

print(titanic_data.head())

titanic_data.embark_town.value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel("Emabark Town")
plt.ylabel("Number of Passengers")

plt.show()

titanic_data.embark_town.mode()

titanic_data.embark_town.fillna("Southampton",inplace=True)

# Categorical Data Encoding

## One hot encoding

titanic_data= sns.load_dataset("titanic")

titanic_data= titanic_data[["sex","class","embark_town"]]
print(titanic_data.head())
temp= pd.get_dummies(titanic_data['sex'])
print(temp.head())

print(pd.concat([titanic_data['sex'], temp]))

## Label Encoding

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

le.fit(titanic_data["class"])

titanic_data['le_class']=le.transform(titanic_data['class'])
print(titanic_data.head())


