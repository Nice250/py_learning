lst_ruku=[]
#商品入库
for i in range(5):
    goods=input('请输入商品信息:')
    lst_ruku.append(goods)
#打印商品清单
for i in range(5):
    print(lst_ruku[i])
#添加到购物车
lst_gouwuche=[]
while 1:
    flag=False#标志变量
    x=input('请输入要购买的商品编号')
    if x=='q':
        break
    for item in lst_ruku:
        if x==item[0:4]:
            lst_gouwuche.append(item)
            print('已成功添加')
            flag=True
            break
    if not flag:
        print('商品不存在')
#打印购物车
lst_gouwuche.reverse()
for item in lst_gouwuche:
    print(item)

