'''
基本结构：---->  for 循环变量 in 遍历对象:
'''

for i in 'hello':
    print(i)

for x in range(1,10):#range(a,b)表示产生一个a到b的整数序列，包含a，但不包含b
    print(x)


# testjd


#寻找100~999之间的水仙花数
for i in range(100,1000):
    gewei=i%10
    shiwei=i//10%10
    baiwei=i//100

    if i==gewei**3+shiwei**3+baiwei**3:
        print(i)

