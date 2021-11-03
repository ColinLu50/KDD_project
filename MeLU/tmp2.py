
a = [1, 2]
b = ['a', 'b']
c = ['11', '22']

l = list(zip(a, b, c))

print(l)

a, b, c= zip(*l)

print(a)
print(b)
print(c)