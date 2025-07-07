#一、try......expect:.....结构

'''
try:
    ........................--->可能出现异常的代码
expect 错误的捕获类型1:
    ........................--->成功捕获后执行的代码
expect 错误的捕获类型2:
    ........................
else:
    ........................--->没有异常时执行的代码
finally:
    ........................--->无论是否产生异常都要执行的代码
'''
'''
常见的捕获类型有：SyntaxError(语法错误)/ZeroDivisionError(除数为0）/TypeError(类型不正确)/ValueError(类型正确但不合法)/NameError(变量或函数名未提前声明)/KeyError(字典取值时键不存在)/IndexError(索引超出范围)

'''
try:
    num1 = int(input('请输入一个整数:'))
    num2 = int(input('请输入另一个整数:'))
    result = num1/num2
    print('结果为:',result)
except ZeroDivisionError:
    print('除数不能为0')
except ValueError:
    print('不能将字符串转成整数')
except TypeError:
    print('类ing')
except BaseException:
    print('未知异常')
else:
    print('程序无错误')
finally:
    print('程序结束')

#二、raise--->由于实际情况的需要，人为抛出异常
try:
    x = int(input('请输入一个正数'))
    if x <= 0:
        raise Exception('这不是正数')#Exception为自定义异常类型
    else:
        print('您输入的正数为:',x)
except Exception as e:#e可以改变为其他变量名
    print(e)
