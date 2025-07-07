def fun(n):
    if n<0:
        return -1
    elif n==1:
        return 1
    else:
        lst=[2,8]
        for i in range (1,n):
            lst.append(lst[-1]+lst[-2])
            return lst[-2]%lst[-1]
print(fun(2))

import random

lst=[random.randint(1,100) for i in range(10)]
max=
for i in range(10):
    if lst[i]<lst[i+1]:
        max=lst[i+1]