'''字典是可变数据类型，其中key与value对应，key不可重复必须为不可变数据类型（例如元组）'''
'''注：字典本身是无序数据类型，只是经过python3.11更新过后，为方便显示为有序'''
#字典的创建
#1、直接创建
d={10:'cat',20:'dog',15:'zoo'}
print(d)
#2、使用zip函数将key与value对应
lst1=[10,24,36]
lst2=['a','b','c']
dzip=zip(lst1,lst2)#此时的d1是一个zip对象,需要将其转化为字典类型
print(type(dzip))
d1=dict(dzip)#转成字典型
print(d1)
#3、使用参数创建字典
d=dict(cat=10,dog=20,zoo=12)
print(d)
print('------------------------------------------')
#4、字典生成式
#要求-->值为1~100之间的随机数，键为0，1，2，3
import random
#(1)直接对应
d={key:random.randint(1,100) for key in range(4)}
print(d)
#(2)利用zip函数将key列表与value列表对应
lstkey=list(range(4))
lstvalue=list(random.randint(1,100) for i in range(4))#列表生成式
d={key:value for key,value in zip(lstkey,lstvalue)}
print(d)

print('----------------------')
#访问字典元素
d={10:'cat',20:'dog',15:'zoo'}
#(1)d[key]
print(d[20])
#(2)d.get(key)
print(d.get(20))
'''二者是有区别的，当访问的key不存在时第一种方法会报错，第二种会提示None'''
print('---------------------------------------------------')

#字典的遍历
#(1)将key与value作为整体(键值对)遍历
for i in d.items():
    print(i)#i代表key与value组成的一个元组
#(2)分别将key和value遍历
for key,value in d.items():
    print(key,value)
print('-------------------')

#字典的操作方法
#(1)添加元素
d1={11:'hhh',14:'lk',22:'sa'}
d1[23]='jih'
print(d1)
print('--------------------')
#(2)查看值
values=d1.values()
print(values,type(values))#此时为字典类型
print(list(values))#转成列表类型
print(tuple(values))#转成元组类型
print('-------------------------------')
#(3)查看键值对
print(d1.items())#此时为字典类型
lst=list(d1.items())#此时转成列表类型 dict_items([(11, 'hhh'), (14, 'lk'), (22, 'sa'), (23, 'jih')])
print(lst)
print('---------------------')
#(4)删除元素
d1.pop(14)#pop()先把key对应值取出来，再删除
print(d1)
print('--------------------------')
#清空字典
#d1.clear()

#合并字典元素符(注意key不要重复)
d1={1:'a',2:'b',3:'c'}
d2={4:'d',5:'e',6:'f'}
total_d=d1|d2
print(total_d)
print('--------------------------------------------------------------')

#例题
d1={1:'a',2:'b',3:'c'}
print(d1)
d2=d1#令d2=d1即代表两个字典指向同一块内存空间，改变其中一个另一个也会改变
d1[2]='k'
print(d2)
