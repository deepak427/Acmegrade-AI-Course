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



