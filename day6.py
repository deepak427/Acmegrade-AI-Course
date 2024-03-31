# String Manipulation

val="a, b , gudio"
print(val.split(","))

pieces=[x.strip() for x in val.split(",")]
print(pieces)

first, second, third=pieces
print(first+"::"+second+"::"+third)

print("::")