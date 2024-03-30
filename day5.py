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

data=pd.DataFrame([[1,4,5.6],[2,np.nan,5],[np.nan,np.nan,np.nan],[1,2,3],[1,2,3]])
print(data.dropna(how='all'))
data[5]=np.nan # data[0] is first coulmn
print(data)
print(data.dropna(thresh=2))
print(data.fillna(0))
print(data.fillna({0:1,1:2.22222}))
print(data.fillna(method='ffill',limit=2))

# Duplicate data

print(data.duplicated())
print(data.drop_duplicates())
data['v1'] = range(5)
print(data.drop_duplicates(subset=[0])) # subset=[0] means that only the values in the first column (indexed as 0) will be used to identify duplicates. 

print(string_data.replace(['Deepak', 'Hero'],['Deepak Singh', 'Hiro']))
print(string_data.replace({'Hiro': 'Hero','Deepak Singh': 'Deepak'}))

data=pd.DataFrame(np.arange(12).reshape(3,4),
                  index=['Ohio','Color','New'],
                  columns=['one','two','three','four'])
print(data)

def transform(x):
    return x[:4].upper()

data.index=data.index.map(transform)
print(data)

data=data.rename(index=str.title, columns=str.upper)
data=data.rename(index={'Ohio':"Indiana"}, columns={'THREE':"FIVE"})
print(data)