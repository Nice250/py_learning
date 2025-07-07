#(1)isdigit()十进制的阿拉伯数字(1,2,3)
print('123'.isdigit()),#True
print('一二三壹贰'.isdigit())#False
print('----------------------------------')
#(2)isnumeric()所有字符都是数字,不能识别二进制
print('0b1010'.isnumeric())#False
print('------------------------------------')
#(3)isalpha()所有字符都是字母(包含中文)
print('你好12abc'.isalpha())#False
print('你好一二三壹贰abc'.isalpha())#True
print('----------------------------')
#(4)isalnum()所有字符都是数字或字母
print('你好一二三壹贰abc132'.isalnum())#True
print('------------------------')
#(5)islower()所有字符都是小写，注：汉字既是小写又是大写
print('abc一二'.islower())#True
print('----------------------------')
#(6)isupper()所有字符都是大写
print('ANBC一二'.isupper())#True
print('---------------------------------')
#(7)istitle()仅首字母大写
print('HelloWorld'.istitle())#False
print('Hello World'.istitle())#True
print('Hello world'.istitle())#False
print('--------------------------------------')
#(8)isspace()判断是否为空白字符



