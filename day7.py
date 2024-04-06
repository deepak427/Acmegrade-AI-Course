import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# loading dataset
data = sns.load_dataset("mpg")
print(data.head())

data['force']=data['weight']*data['acceleration']
print(data.head())

data['force'].plot.hist(bins=50)
plt.show()
data['force'].plot.density()
plt.show()

#Regplot and Pairplot

sns.regplot(x = "mpg",
            y = "acceleration",
            data = data,
            dropna = True)
plt.show()

sns.pairplot(data, hue ='weight')
plt.show()

#sns.catplot # specify rows and columns
#Box plot

#Data aggeragation and group operations
#df.groupby
#grouped.mean