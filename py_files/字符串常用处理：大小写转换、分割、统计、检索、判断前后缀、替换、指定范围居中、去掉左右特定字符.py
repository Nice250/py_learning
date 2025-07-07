#大小写转换s1.lower()/s1.upper()
s1='HELLOWORLD'
new_s1=s1.lower()
print(new_s1)
print('---------------------------------------------')
s2='apple'
new_s2=s2.upper()
print(new_s2)
print('---------------------------------------------------')

#分割s.split('x')
e_mail='2856317785@qq.com'
lst=e_mail.split('@')#将字符串按某个特点字符左右分隔，得到一个列表
print('账号:',lst[0],'后缀:',lst[1])

#统计某个字符在字符串中出现的次数s.count('x')
s1='apple'
num=s1.count('p')
print(num)
print('--------------------------------------------')

#检索操作s1.index('x')/s1.find('x')
s1='apple'
print(s1.find('a'))
print(s1.index('a'))
print(s1.find('M'))#区别：找不到时输出-1
#print(s1.index('M'))#找不到时会报错
print('----------------------------------')

#判断前缀、后缀s1.startwith('x')/s1.endwith('x')
s1='hello world'
print(s1.startswith('hel'))
print(s1.endswith('rld'))
print('------------------------------------')

#替换s1.replace(old,new，count)
s1='我今年18岁，我是大学生'
s2=s1.replace('我','你',1)#最后一个参数代表替换次数，不写的话就默认是去全部替换
print(s2)
print('-------------------------------------')

#指定范围居中s1.center(width,fillwith)
s1='hello'
print(s1.center(20,'*'))
print('-------------------------------------')

#去掉字符串左侧/右侧的指定字符s.strip('x')
s='*/world*//'
print(s.strip('/*'))#写作print(s.strip('/*'))也可，与顺序无关
print(s.lstrip('/*'))#只去掉左边
print(s.rstrip('/*'))#只去掉右边
