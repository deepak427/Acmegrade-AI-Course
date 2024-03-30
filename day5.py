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