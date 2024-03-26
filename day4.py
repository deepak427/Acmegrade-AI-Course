import numpy as np

print(np.zeros((3,4)))
print(np.ones((3,4)))

arr1=np.array([1,2,3],dtype=np.float64)
arr2=np.array([1,2,3],dtype=np.int32)

float_array=arr2.astype(np.float64)
print(float_array.dtype)

arr=np.array([[1,2,3],[4,5,6]])
print(arr)
print(arr*arr)
print(arr-arr)
print(arr2>arr1)

arr=np.arange(15).reshape((5,3))
print(arr)

print(arr.T)
print(np.dot(arr,arr.T))

print(arr.T@arr)

# Pandas

import pandas as pd

# obj=pd.Series([4,7,-5,3])
# print(obj)

# obj2=pd.Series([4,7,-5,3],index=["a","b","c","d"])
# obj2["d"]=6
# print(obj2["a"])
# print(obj2[obj2>0])

# sdata={'Ohio':35000,"Texas":"71000"}
# obj3=pd.Series(sdata)
# print(obj3.to_dict())

# data={"states":["Ohio","Ohio","Ohio","Neveda"],
#       "year":[2000,2001,2002,2004],
#       "pop":[1.5,1.7,3.6,2.4]}
# frame=pd.DataFrame(data, columns=["pop","states","year"])
# print(frame.head())
# print(frame.loc[[1],["pop"]])
# print(frame[frame["year"]>2000])