import numpy as np
import pandas as pd

# unique values and counts
obj=pd.Series(["a","b","c","d"])
print(obj)
print(obj.unique())
print(obj.value_counts())

#loading data

print(pd.read_csv("sample.csv",header=None))
names=[0,1,2,3,4,5,6,7,8,9]
print(pd.read_csv('sample.csv',names=names,index_col=3))

# pd.read_csv("sample.txt",sep="\s+")

result = pd.read_csv("sample.csv",skiprows=[0,2,3])
result.to_csv("out.csv")

# Data preprocessing

string_data=pd.Series(["Deepak",np.nan,None,"Hero"])
print(string_data.isna())
print(string_data.dropna())

data=pd.DataFrame([[1,4,5.6],[2,np.nan,5],[np.nan,np.nan,np.nan]])
print(data.dropna(how='all'))
data[5]=np.nan # data[0] is first coulmn
print(data)
print(data.dropna(thresh=2))
print(data.fillna(0))
print(data.fillna({0:1,1:2.22222}))
print(data.fillna(method='ffill',limit=2))