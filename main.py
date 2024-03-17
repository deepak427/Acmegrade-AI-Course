# data types

x=5
print(x, type(x))

y=10.5
print(y, type(y))

z='Deepak Singh'
print(z, type(z))

tup_a=(4,5,6)
tup_b=tuple(z)
print(tup_a, type(tup_a))
print(tup_b, type(tup_a))

#slicing and indexing

print(tup_b[0:3])
print(tup_b[:])
print(tup_b[::2])

# nested tuple

nested_tup=((4,5,6),(7,8))
print(nested_tup[0])

print(tup_a+tup_b+nested_tup)

print(tup_a.count(4))

#List

