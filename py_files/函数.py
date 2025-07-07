'''
def 函数名 (形参a,形参b,......):
    函数体
    ...
    ...
    ...
    return ... #返回值，没有的话可以不写
'''

def get_sum(num):
    sum=0
    for i in range(1,num+1):
        sum+=i
    print(f'1~{num}之间的累加和为:{sum}')

get_sum(5)#函数的调用 函数名(实参)
get_sum(6)

#位置传参与关键字传参
def happy_birthday(name,age):
    print(name+str(age)+'生日快乐')

happy_birthday('刘',19)#位置传参(参数数量和顺序要对应)
happy_birthday(name='刘',age=19)#关键字传参(顺序可以不同)
happy_birthday('刘',age=19)#混合使用，注意要遵循:位置参数在前，关键字参数在后

#默认值参数
def fun(a,b=10):
    pass
'''
def
#当函数的形参既有位置参数又有默认值参数时，应遵循:位置参数在前，默认值参数在后
def fun(a=10):
    pass
'''

#个数可变的位置参数
def fun (*x):#在参数前加一颗*
    print(type(x))#形参是一个元组类型
    for item in x:
        print(item)

fun(1,2,3,4)
lst1=['a','b','c','d']
fun(*lst1)#当一个列表中的所有元素作为实参传入时，需在列表名前加*，将列表解包，从而将每个元素传入函数

#个数可变的关键字参数
def fun2(**y):#在形参前加**
    print(type(y))#形参是一个字典(关键字为键，实参为值)
    for key,value in y.items():#字典的遍历
        print(key,'----',value)
fun2(name='刘',age=19)#传参时必须关键字传参
dic={'name':'刘','age':18}
fun2(**dic)#当一个字典中的键、值作为实参传入时，需在字典名前加**，将字典解包，从而将键、值传入函数

#函数有多个返回值时，可以通过解包赋值
def get_sum(x):
    sum=0
    for i in range(1,x+1):
        sum+=i
    return x,sum
a,b=get_sum(5)
print(a,b)

#局部变量与全局变量
'''
(1)当全局变量与局部变量名相同时，局部变量优先
(2)可以使用global在函数内进行全局变量的声明，注意声明和赋值要分开
'''

#匿名函数
'''
指的是没有名字的函数 ，这种函数只能使用一次，一般是在函数体只有一句代码或只有一个返回值时使用
语法结构:
    result=lambda 形式参数列表:表达式
'''
#eg.1
s=lambda a,b:a+b
print(s(3,4))

#eg.2
lst=['a','b','c','d' ,'e','f']
for i in range(len(lst)):
        result=lambda x:x[i]
        print(result(lst))

#eg.3
student_scores=[{'小红':95},{'小兰':86},{'小美':98},{'小刚':65}]
student_scores.sort(key=lambda x:list(x.values())[0],reverse=True)#详见列表排序&字典的值的取法
print(student_scores)























