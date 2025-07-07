'''集合是无序的不重复的可变序列，集合只能存储不可变数据类型(列表，字典不能作为元素）。'''
#集合的创建
#(1)使用{}直接创建
s={10,20,30,40}
print(s)
#(2)使用set()创建
s=set('helloworld')
print(s)#结果为{'w', 'l', 'r', 'h', 'd', 'o', 'e'}说明元素不重复且无序
s=set([1,2,3,4])#元素是列表中的整数，为不变数据类型，故可以创建
print(s)
print('--------------------')
'''集合是序列的一种，故序列相关操作适用于集合，max(),min(),len(), a in s等等'''
#(3)集合生成式

s={i for i in range(10)}#直接生成
print(s)
s={i for i in range(10) if i%2==0 }#条件生成式
print(s)
print('--------------------------------')


#集合操作符
A={1,2,3,4,5}
B={3,4,5,6,7}
#(1)交集
print(A&B)
print('--------------------')
#(2)并集
print(A|B)
print('--------------------')
#(3)差集
print(A-B)#A有且B没有
print(B-A)#B有且A没有
print('--------------------')
#(4)补集
print(A^B)
print('--------------------')

#集合相关操作
s={10,20,30}
#添加元素
s.add(100)
print(s)
print('----------------------')
#删除元素
s.remove(20)
print(s)
#清空
#s.clear()
print('------------------------')
#集合的遍历
#方法一（直接输出）
for i in s:
    print(i)
#方法二(利用enumerate()按指定序号输出）
for index,item in enumerate(s,start=1):
    print(index,item)

