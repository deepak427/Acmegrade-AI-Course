#data types

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

a_list=[2,3,7,None]
print(a_list, type(a_list))

print(list(range(20)))

a_list.append('append')
a_list.insert(1,'first')
print(a_list)

a_list.pop(1)
a_list.remove(None)
print(a_list)

x=[4,None,"foo"]
x.extend([7,8,(2,3)])
print(x)

a=[1,2,3,4,1,2,5,6,8,2,4,5]
a.sort()
a.sort(reverse=True)
print(a)

b=['saw','do','Hello','sound']
b.sort(key=len)
print(b)

b[0:2]=["foo","dipu"]
print(b)
