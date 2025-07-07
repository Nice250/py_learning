'''
正则表达式学习：https://blog.csdn.net/LLLLQZ/article/details/118278287
'''

import re #导入re模块
#re.match(pattern,待更改字符串,修饰词)从头开始匹配，匹配到就返回数据，没有就返回None
pattern='\d\.\d+'#表示x.xxxxxx数字
s1='3.1415,361.0,1.25'
match=re.match(pattern,s1,re.I)#从头开始匹配，忽略大小写
print(match)

print('匹配的起始位置',match.start())
print('匹配的结束位置',match.end())
print('匹配的区间',match.span())
print('待匹配的字符串',match.string)
print('匹配的数据',match.group())
print('------------------------------------------')
#re.search(pattern,待更改字符串,修饰词)对整个字符串匹配，但是只匹配一个数据,匹配到就返回单个数据，没有就返回None
pattern='\d\.\d+'
s2='563,3.1415,361.0,1.25'
match2=re.search(pattern,s2)
print(match2)
print('匹配的数据',match2.group())
print('--------------------------------------------------------')
#re.findall(pattern,待更改字符串,修饰词)对整个字符串匹配，匹配所有符合正则表达式数据,匹配到就返回数据列表，没有就返回空列表
pattern1='\d\.\d+'
s2='563.0,3.1415,361.0,1.25'
match3=re.findall(pattern,s2)
print(match3)
print('--------------------------------------------------------')
#sub(pattern,替换字符,待更改字符串）
pattern2='黑客|破解|反爬'
s3='我是黑客，我想学习Python,想破解一些VIP视频,可以实现无底线反爬吗?'
new_s=re.sub(pattern2,'***',s3)
print(new_s)
print('----------------------------------------')
#split(pattern,待更改字符串)
pattern3='[@.]'
s4='2856317785@qq.com'
lst=re.split(pattern3,s4)
print('账号：%s,邮箱平台：%s,域名：%s' % (lst[0],lst[1],lst[2]))#格式化输出
print('--------------------')

'''注：若从第一个字符劈分，则得到的列表第一个元素为空'''





