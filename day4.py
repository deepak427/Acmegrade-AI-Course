# Pandas

import pandas as pd

obj=pd.Series([4,7,-5,3])
print(obj)

obj2=pd.Series([4,7,-5,3],index=["a","b","c","d"])
obj2["d"]=6
print(obj2["a"])
print(obj2[obj2>0])

sdata={'Ohio':35000,"Texas":"71000"}
obj3=pd.Series(sdata)
print(obj3.to_dict())

data={"states":["Ohio","Ohio","Ohio","Neveda",],
      "year":[2000,2001,2002,2004],
      "pop":[1.5,1.7,3.6,2.4]}
frame=pd.DataFrame(data)
print(frame)