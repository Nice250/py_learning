'''
@part1.数学函数
'''
import random

print(abs(-100))#绝对值
print(divmod(13,4))#计算x和y的商和余数，结果为元组类型

print(max([5,1,4,89,988,9,6,56]))#对列表按数字排序
print(min('AaBbkH'))#对字符串按字母排序时小写字母优先级大于大写字母
print(max('AaBbkZz'))
print('---------------------------------------')

print(sum([2,3,4]))#求和
print('-----------------------------------')

print(pow(2,3))#幂的计算
print('----------------------------------------------')
print(round(3.1415,2))#四舍五入round(数据, 保留位数)
'''若保留位数为负数，则对小数点前的数据进行四舍五入'''


'''
@part2.迭代函数 下面代码中的iter均指可迭代对象(python中除了数字类型，基本上都是可迭代对象）
'''
#zip(iter1,iter2)#将iter1,iter2打包成元组，返回一个可迭代对象
x=list(range(6))
y=list(random.randint(1,10) for i in range(5))#生成5个【1，10】之间的随机数
print(x)
print(y)
zipobj=zip(x,y)#zip连接时以短的为主
print(list(zipobj))
print('------------------------------------')

#enumerate(iter,start=...)同时获取列表的元素和序号
lst1=['ac','bf','dfs']
enum=enumerate(lst1,start=1)
print(list(enum))
print('---------------------------------------')

#all(iter)判断列表是否所有元素布尔值都为1，相当于交集
lst2=[1,2,3,'ab',5]
lst3=[1,2,0,5,4]
print(all(lst2))
print(all(lst3))

#any(iter)判断所有元素布尔值都为0，相当于并集,全为0，结果才为0
lst2=[1,2,3,'ab',5]
lst3=[0,(),[],{}]
print(any(lst2))
print(any(lst3))
print('--------------------------')

#next(iter)获取迭代器的下一个元素
lst4=['ac','bf','dfs']
enum=enumerate(lst4,start=1)
print(next(enum))
print(next(enum))
print('-------------------------------')

#filter(fun,iter)通过函数过滤序列(将序列带入函数，返回值为TRUE的存入迭代器中)并返回一个迭代器对象
def fun (x):
    return x%2==1
filt=filter(fun,range(10))
print(list(filt))#迭代器对象无法直接看到结果，需要转成列表

#map(fun,iter)通过函数function对可迭代对象iter操作，并返回一个迭代器对象,可以代替遍历
def upword(x):
    return x.upper()
n_list=['hello','world','abc']
mp=map(upword,n_list)
print(list(mp))



