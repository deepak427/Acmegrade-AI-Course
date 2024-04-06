import pandas as pd
from matplotlib import pyplot as plt

tips=pd.read_csv('data/sample.csv')
print(tips.head())

import seaborn as sns

tips['Year_new']=tips['Year']+5
print(tips.head())

tips['Year_new'].plot.hist(bins=50)
plt.show()
tips['Year_new'].plot.density()
plt.show()

#sns.regplot
#sns.pairplot