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
scalar.fit(data)

data_scaled=scalar.transform(data)

print(data.head())

sns.kdeplot(data=["age"])

# Min max scalar

from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()
scalar.fit(data)

data_scaled=scalar.transform(data)
print(data.head())



