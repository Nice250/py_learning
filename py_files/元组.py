'''
1、元组是不可变数据类型，有序，可重复
2、元组占用内存小，列表占用内存大
'''
#元组的创建
#法一、直接创建
t=(12,'hello',[2,3,5])
print(t)
t=(1,)#仅有一个元素的元组创建不要忘记","
#利用内置函数tuple
t=tuple(range(1,10))
print(t)
print('---------------------')
#元组的删除
#del t

#元组的遍历
t=(12,'hello',[2,3,5])
#方法一：
for item in t:
    print(item)

print('--------------------')
#方法二
for i in range(len(t)):
    print(i,t[i])

print('---------------------')
#方法三
for index,item in enumerate(t,start=1):
    print(index,item)