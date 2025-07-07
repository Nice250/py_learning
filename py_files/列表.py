'''列表是可变数据类型，元素可重复，有序'''
#创建
lst1=['sa','d',1,556]#使用[]直接创建
print(lst1)
lst2=list('helloworld'[::2])#使用内置函数list创建
print(lst2)
lst3=list(range(1,10,2))
print(lst3)
print('----------------------')
#一般列表生成式 lst=[元素 for i in range（列表元素的个数）]

lst=[item for item in range(1,11)]
print(lst)

lst=[x*x for x in range(1,11)]
print(lst)

import random#引入随机数函数
lst=[random.randint(1,100) for a in range(1,11)]#产生10个1~100之间的随机数
print(lst)

#条件列表生成式
lst=[i for i in range(1,11) if i%2==0]#生成1~10之间的偶数列表
print(lst)

#删除
lst4=[1,2,3,4]
del lst4#使用del函数删除

#列表的遍历
lst5=['hi','you','are',54]

'''1、for循环'''
for item in lst5:
    print (item)
print('--------------------')

'''2、使用for循环，根据range()函数，len()函数--->计算一维列表中元素个数（计算二位列表中的行数），根据索引遍历'''
for i in range(0,len(lst5)):
    print(i,lst5[i])
print('--------------------')

'''3、使用enumerate()函数'''
for index,item in enumerate(lst5,start=1):#start=x表示index起始序号
    print(index,item)

#列表的相关操作
lst6=['ni','ji','mg',5]

#在末尾增加元素 每次只能添加一个元素
lst6.append('fgh')
print(lst6)
print('------------------')

#插入元素
lst6.insert(1,123)#在index索引处添加元素
print(lst6)
print('------------------')

#列表元素的删除
lst6.remove('ji')#使用remove根据对象删除
print(lst6)
lst6.pop(1)#使用pop()根据索引删除
print(lst6)
print('--------------------')

#列表的反向
lst6.reverse()
print(lst6)
print(lst6.reverse())#reverse()是对列表本身进行改变，没有返回值，故为None
print('--------------------')

#列表的拷贝
new_lst=lst6.copy()
print(new_lst)
print('-----------------')

#列表的排序
lst7=list('47842952')
print("原列表",lst7)

'''法一：使用sort()时原列表被改变'''
#升序
lst7.sort()#sort()内不填时，默认升序
print(lst7)
#降序
lst7.sort(reverse=True)
print(lst7)

'''不指定规则时对于字母升序排序时：先排大写，再排小写；对于字母降序排序时：先排小写，再排大写'''
lst8=list("asJjfKMijka")
lst8.sort(key=str.lower)#指定规则：忽略大小写比较
print(lst8)


'''法二：使用sorted(x,key=...,reverse=...)时原列表不变，产生新列表'''
lst9=list("15489652")
print("原列表为",lst9)

new_list2=sorted(lst9)#默认升序排序
print(new_list2)

new_list3=sorted(lst9,reverse=True)#降序排序
print(new_list3)

print('---------------------------------')
print('---------------------------------')

'''二维列表'''
#二维列表的创建(不要忘记逗号）
lst=[
    [13,'hello',98],
    ['asd',555,'ok'],
    [89,5,14],
    [58,'d','s',89]
]
print(lst)
print('----------------------------')
print(lst[0][0])#打印某个特地数据
print(len(lst))#实际计算的是行数
print(len(lst[0]))#打印列数
print('------------------------------')
#二维列表的遍历
for row in lst:#行
    for item in row:#列
        print(item,end=' ')
    print()#每打印完一行都要换行
print('------------------------')
'''另一种麻烦方法，但容易理解for i in range(0,len(lst)):#行
    for j in range(0,len(lst[i])):#列
        print(lst[i][j],end=' ')
    print()#每打印完一行都要换行'''
print('------------------------------')
#二维列表生成式
lst=[[j for j in range(5)]for i in range(4)]#4行 5列
print(lst)
