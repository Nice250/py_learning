row=eval(input('请输入行数:'))

while row %2 == 0:
    row=eval(input('请输入正确的行数:'))

#上半行的输入
for i in range(1,(row+1)//2):
    print(' '*((row+1)//2-i),end='')
    print('*'*(2*i-1))
    i=i+1

for i in range(1,(row+1)//2+1):
    print(' '*(i-1),end='')
    print('*'*(-2*i+row+2))
    i=i+1



'''
--*
-***
*****
-***
--*

------*
-----***
----*****
'''

