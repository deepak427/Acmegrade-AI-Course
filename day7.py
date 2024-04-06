import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# loading dataset
data = sns.load_dataset("mpg")
print(data.head())

data['force']=data['weight']*data['acceleration']
print(data.head())

data['force'].plot.hist(bins=50)
# plt.show()
data['force'].plot.density()
# plt.show()

#Regplot and Pairplot

sns.regplot(x = "mpg",
            y = "acceleration",
            data = data,
            dropna = True)
# plt.show()

# sns.pairplot(data, hue ='weight')
# plt.show()

#sns.catplot # specify rows and columns
#Box plot

#Data aggeragation and group operations

print(data.groupby(["name", "model_year"])["horsepower"].mean().head())
print(data.groupby(["model_year"])["name"].count().head())
print(data.groupby("name", dropna=False).size())

for name, group in data.groupby("model_year"):
    print(name)
    print(group)

print(data.groupby(["name", "model_year"])["horsepower"].agg(["mean","std","count"]).head())
print(data.groupby(["name", "model_year"])["horsepower"].agg([("average","mean"),("std",np.std)]).head())
