if True:
    print(666)
if False:
    print(666)
#若if后连接的对象的布尔值为True（1）/not False，则执行下方语句
#若if后连接的对象的布尔值为False（0）/not True]，则不执行
print('----------------')
x='False'
if x:
    print(555)
    ''' 此处‘False’是一个非空字符串，它对应的布尔值为True，故执行'''

y = eval('False')#eval去引号
if y:
    print(222)
    '''此处经过eval后为保留字False，其布尔值是False，故不执行'''
print('----------------')

#二重比较
print('yes' if 1>2 else 'no')

print('-----------------')

#多分支结构
score=eval(input('请输入成绩：'))
if 90<score<=100:
    print('优秀')
elif 60<score<=90:
    print('良好')
elif 0<score<=60:
    print('不及格')
else:
    print('???')

print('-----------------')

#模式匹配
score=input('请输入成绩等级：')
match score:
    case 'A':
        print('优秀')
    case 'B':
        print('良好')
    case 'C':
        print('不及格')

print('-----------------')







