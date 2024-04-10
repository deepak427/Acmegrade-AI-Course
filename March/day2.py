# Conditional statements

x=10

if x>=10:
    print('Yes')
else:
    print('No')

# loops

list_c=[10,20,30,40,50]

for i in list_c:
    print(i)

x=5

while x>=1:
    print(x)
    x=x-1

# dictionary
    
d1={"a":"some value here", "b":[1,2,3,4], 1:"Hello"}
print(d1,type(d1))
print(d1.keys(), d1.values(), d1.items())

d1.update({"b":"foo","4":12})
print(d1)

words=['apple','bat','bar','atom','book','cook']

by_letter={}

for word in words:
    letter=word[0]
    if letter not in by_letter:
        by_letter[letter]=[word]
    else:
        by_letter[letter].append(word)

print(by_letter)

# sets

a={1,2,3,4,5}
b={3,4,5,6,7,8}

print(a.union(b))
print(a|b,a&b)

# Comprehension

strings=['a','as','bat','car','dove','python']

c=[x.upper() for x in strings if len(x)>2]

print(c)