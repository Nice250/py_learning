fp=open('text.txt','w')#打开文件
print('你好',file=fp)#输出内容到文件
fp.close()#关闭文件

#练习1
name=input('请输入您的姓名:')
age=input('请输入您的年龄:')
mos=input('请输入您的座右铭:')
print('--------自我介绍--------')
print('姓名：'+name)
print('年龄：'+age)
print('座右铭：'+mos)