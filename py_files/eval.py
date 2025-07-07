#eval函数可以去掉字符串中的引号
s='2.4+3.2'
print(s,type(s))
s1=eval(s)
print(s1,type(s1))

#eval常与input连用，用于获取数值，因为input会默认输入为字符串
age=eval(input('请输入年龄:'))
print(age,type(age))

test