# Nested list comprehension

all_data=[["Deepak", "Bhaiya", "Didi"],
          ["Mummy", "Papa"]]

print(all_data)

names_of_interest=[]

for names in all_data:
    enough_as=[name for name in names if name.count("a")>=2]
    names_of_interest.extend(enough_as)

print(names_of_interest)

result=[name for names in all_data for name in names if name.count("a")>=2]
print(result)

# Functions

def my_function(x,y):
    return x+y
print(my_function(10, 20))

def my_function(x,y,z=1.4):
    if z>1:
        return z*(x+y)
    else:
        return z/(x+y)
    
print(my_function(5,6))
print(my_function(5,6,z=0.7))
print(my_function(5,6,0.7))
print(my_function(x=5,y=6,z=0.7))

def func():
    global a
    a=[]
    for i in range(5):
        a.append(i)

func()
print(a)

def f():
    a=5
    b=6
    c=7
    return a,b,c

a,b,c=f()
print(a,b,c)

states=[" Alabama ","Georgial","south carolina##","West virginia?"]

import re

def clean_strings(strings):
    result=[]
    for value in strings:
        value=value.strip()
        value=re.sub("[!#?]","",value)
        value=value.title()
        result.append(value)
    return result

print(clean_strings(states))

# lambda functions

eqiv= lambda x,y:x*y*2
print(eqiv(20,30))

def apply_to_list(some_list,f):
    return [f(x) for x in some_list]

ints=[4,2,5,2,5]

print(apply_to_list(ints, lambda x:x*2))

strings=["foo","card","bar","aaaa","abab"]
strings.sort(key=lambda x:len(set(x)))

print(strings)

# errors and exception handling

def attempt_float(x):
    try:
        return float(x)
    except:
        return x

print(attempt_float("9a"))

# Numpy

import numpy as np

my_arr=np.arange(1_000_000)
print(my_arr[1:10])

my_list=list(range(1_000_000))
print(my_arr[1:10])

data=np.array([[1.5,0.1,3],[0,-3,6.5]])
print(data*10)

print(data+data)
print(data.shape)
print(data.dtype)
print(data.ndim)

data1=[5,6,3,7,2,2]

arr1=np.array(data1)
