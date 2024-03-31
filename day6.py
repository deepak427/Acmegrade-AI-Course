import pandas as pd
import numpy as np

# String Manipulation

val="a, b , gudio"
print(val.split(","))

pieces=[x.strip() for x in val.split(",")]
print(pieces)

first, second, third=pieces
print(first+"::"+second+"::"+third)

print("::".join(pieces))

# Data Wrangling

data=pd.Series(np.random.uniform(size=9),
               index=[["a","a","a","b","b","b","c","c","c"],
                      [1,2,3,1,2,3,1,2,3]])
print(data)
print(data['b'][3])
print(data["a":"c"])
print(data.loc[["a","c"]])

print(data.unstack())

frame=pd.DataFrame(np.arange(12).reshape((4,3)),
                   index=[["a","a","b","b"],[1,2,1,3]],
                   columns=[["Ohio","Ohio","Coronada"],["Green","Red","Green"]])
print(frame)

frame.index.names=["key1","key2"]
frame.columns.names=["state","color"]
print(frame)
print(frame.index.nlevels)

# Combining df

df1=pd.DataFrame({"key":["b","b","a","c","a","a"],
                  "data1":[1,2,3,4,5,6]})
df2=pd.DataFrame({"key":["b","r","c"],
                  "data2":[12,34,34]})

df3=pd.DataFrame({"lkey":["b","b","a","c","a","a"],
                  "data1":[1,2,3,4,5,6]})
df4=pd.DataFrame({"rkey":["b","r","c"],
                  "data2":[12,34,34]})

print(pd.merge(df1,df2))
print(pd.merge(df3,df4, left_on="lkey", right_on="rkey"))

print(pd.merge(df1,df2, how="outer"))
print(pd.merge(df3,df4, left_on="lkey", right_on="rkey", how="outer"))

print(pd.merge(df1,df2, how="left"))

# 

