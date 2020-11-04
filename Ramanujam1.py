def cube(a):
    ''' check if a number is a perfect cube '''
    for i in range(1, a+1):
        if i**3 > a:
            break
        elif i**3 == a:
            return a

def ramanujam(x):
    ''' checks all numbers in a given range if they are a Hardy-Ramanujan number'''
    a = []
    for i in range (1, int(x**0.5)):
        y = x - i**3
        if cube(y):
            a.append((i, int(round(y**(1/3),0))))
        else:
            pass
        if len(a) == 2 and a[0][0] != a[1][1]:
            for p in a:
                print('{}^3 + {}^3 = {}'.format(p[0], p[1], x))
            return print('{} is a Hardy-Ramanujan number!'.format(x))

for k in range(1, 100000):
    ramanujam(k)
