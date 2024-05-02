# Data Preprocessing
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

## Feature Scaling

### Standarization

from sklearn.preprocessing import StandardScaler

data = sns.load_dataset("iris")

scalar=StandardScaler()
print(data.head())

data_to_scale=data[["sepal_length","sepal_width"]]
data_scaled=scalar.fit_transform(data_to_scale)
scaled_df = pd.DataFrame(data_scaled, columns=data.columns[:2])

print(scaled_df)

# sns.kdeplot(data["sepal_length"])
# plt.show()

# Min max scalar

from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()

data_scaled=scalar.fit_transform(data_to_scale)
print(data_scaled[:5])

# Handling Missing Values

data=data[["petal_length","petal_width",]]
print(data.isnull().mean())

median=data.petal_length.median()
print(median)
data["petal_length"]=data.petal_length.fillna(median)

# Frequent Category Imputation

titanic_data= sns.load_dataset("titanic")

titanic_data= titanic_data[["embark_town","age","fare"]]
print(titanic_data.head())
print(titanic_data.isnull().mean())

titanic_data.embark_town.value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel("Emabark Town")
plt.ylabel("Number of Passengers")

# plt.show()

titanic_data.embark_town.mode()

titanic_data.embark_town.fillna("Southampton",inplace=True)

print(titanic_data.head())

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

# Discretization

import warnings
warnings.filterwarnings("ignore")

diamond_data= sns.load_dataset("diamonds")
print(diamond_data.head())

# sns.distplot(diamond_data['price'])
# plt.show()

price_range=diamond_data['price'].max()-diamond_data["price"].min()

lower_interval= int(np.floor(diamond_data["price"].min()))

upper_interval= int(np.ceil(diamond_data["price"].max()))

interval_length= int(price_range/10)

print(lower_interval, upper_interval, interval_length)

total_bins=[i for i in range(lower_interval, upper_interval + interval_length, interval_length)]
print(total_bins)

bin_labels=['Bin_no' + str(i) for i in range(1, len(total_bins))]
print(bin_labels)

diamond_data["price_bins"] = pd.cut(diamond_data["price"], bins=total_bins
                                    , labels=bin_labels, include_lowest=True )

print(diamond_data.head())

diamond_data.groupby("price_bins")["price"].count().plot.bar()
plt.show()
# Handling outliers

