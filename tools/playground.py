xs = [('a','b','c','d'), ('a','b','c','d'), ('a','b','c','d')]

ys = {}
for (a, _, _, d) in xs:
    ys = {**ys, **{a: d}}
print(ys)


print({a: d for (a, _, _, d) in xs})