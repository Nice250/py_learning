#多类型数据的输出
#(1)使用占位符
name='马冬梅'
age=18
score=95.5
print('姓名：%s,年龄：%d,成绩：%.1f' % (name,age,score))

#(2)f-string
print(f'姓名：{name},年龄：{age},成绩：{score}')

#(3)使用format
print('姓名：{0},年龄：{1},成绩：{2}'.format(name,age,score))#0,1,2为format中的索引位置

#格式化字符串的详细格式
# '''
# (1)引导符号  :
# (2)填充 对齐方式 宽度   <左对齐  >右对齐   ^居中对齐   :x
# (3)三位分隔符
# (4)保留位数/显示宽度
# (5)整数类型：b\d\o\x\X  浮点数类型：e\E\f\%
# '''
s=('helloworld')
print('{0:*<20}'.format(s))#表示左对齐、以*填充、宽度为20

s1=4851231498.2564#三位分割符，从个位开始，向左三位一逗号
print('{0:,}'.format(s1))

s=4.1254
print('{0:.3f}'.format(s))#表示浮点数的小数点后保留位数

s1='abcdefgh'
print('{0:.3}'.format(s1))#表示字符串的显示宽度,注意不加f

#使用format(value,format_spec)函数
print(format(3.14,'20'))#数值型默认右对齐
print(format('hello','20'))#字符串默认左对齐
print(format('hello','*<20'))#*填充，左对齐，宽度为20
