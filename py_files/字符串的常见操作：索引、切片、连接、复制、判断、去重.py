#字符串的索引
str1='我是哈深好青年活雷锋'
print(str1[0],str1[-10])#0表示从左至右第一个字符
print(str1[9],str1[-1])#-1表示从右至左第一个字符

#字符串的切片[a:b]从a开始到b，但不包括b，且a<b
print(str1[2:7])#从2开始到7，但不包括7
print(str1[-8:-3])
print('-----------------')
print(str1[1:10:2])#通过调整步长实现隔断切割
print('--------------------')
print(str1[::-1])#步长为-1可使字符串逆序
print((str1[-1:-11:-1]))#此语句与上语句作用相同
print('--------------------')


#字符串的连接&复制&判断
str2='我爱'
str3='python'
#(1)字符串的连接
#方法一：+连接
print(str2+str3)
#方法二：以特殊字符连接:'....'.join([s1,s2,......])
print('*'.join([str2,str3]))
print('--------------------')

#(2)*表示将字符串复制n次
print((str2+str3)*3)
str4=2*(str2+str3)
print(str4)
print('--------------------')

#(3)a in b表示判断a是否存在于b中
print('爱' in str2)
print('爱' in str3)
result1='爱' in str2#此时result1，2均代表字符串('True'或者'False')
result2='爱' in str3
print(result1,result2)

result3=int('爱' in str2)
result4=int('爱' in str3)
print(result3,result4)
print('------------------')

#字符串的去重操作
s1='acbddbcdacdabe'
#方法一:直接对元素遍历
new_s1=''
for item in s1:
    if item not in new_s1:
        new_s1=new_s1+item
print(new_s1)

#方法二：使用索引进行遍历
new_s2=''
for i in range(len(s1)):
    if s1[i] not in new_s2:
        new_s2=new_s2+s1[i]
print(new_s2)

#方法三:利用集合元素不重复的特性进行去重
new_s3=set(s1)#集合去重
new_lst=list(new_s3)#转为列表
new_lst.sort(key=s1.index)#由于集合的无序性，故需要按照s1的索引对列表进行排序
print(''.join(new_lst))#将各个字符串连接起来

print('----------------------')


#序列的一些内置函数
str4='cdegeba'
print('长度',len(str4))
print('最大值',max(str4))
print('最小值',min(str4))
print('第一次出现的索引位置',str4.index('g'))
print('对象在序列中出现的次数',str4.count('e'))

