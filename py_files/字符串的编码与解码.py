#编码
s='伟大的中国梦'
scode=s.encode(errors='replace')#默认为utf-8，1个中文占3个字节
print(scode)

scode_gbk=s.encode('gbk',errors='replace')#遇到不能编码的自动替换为？/出错类型还有ignore(忽略）,strict(报错)
print(scode_gbk)#gbk中中文占两个字节
print('-----------------------------------')

#解码
print(bytes.decode(scode,'utf-8'))#编码用什么，解码就用什么
print(bytes.decode(scode_gbk,'gbk'))