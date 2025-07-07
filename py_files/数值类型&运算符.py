#如何查看数据类型
x=350
print('x的数据类型:',type(x))
print('-'*20)

#python允许连续赋值
num1=num2=520
print(num1,num2)
print('-'*20)
#解包赋值（分解字符串）
a,b,c,d='1234'
print(a)
print(b)
print(c)
print(d)

#解包赋值
a,b=10,20#可实现对a，b同时赋值
a,b=b,a#可实现a，b数值交换

#进行比较运算输出布尔值
print(1<2)
print(1==2)
print(1!=2)
print('-'*20)

#逻辑运算符(and-且   or-或   not-非)
print(True and True)#有假则假
print(True and False)
print(False and False)
print('-'*20)
print(True or True)#有真则真
print(True or False)
print(False or False)
print('-'*20)

#常量的定义:使用大写字母和下划线做标识符
PI=3.1415926#不允许修改

#保留一位数字(四舍五入）精度低不建议使用
num3=2.32
num4=2.23
print(round(num3+num4,1))
print('-'*20)

#复数表示
fushu=3+4j#表示方法
print(fushu.real,fushu.imag)#实部用real表示，虚部用imag表示

print('-'*20)

#多行字符串：使用三引号
str1='''我
是
谁
'''
print(str1)
