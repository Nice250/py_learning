#阶乘计算
def fac (n):
    if n==1:
        return 1
    else :
        return n*fac(n-1)
print(fac(5))

#斐波那契数列(1,1,2,3,5,8,13......)
def fb(n):
    if n==1 or n==2:
        return 1
    else:
        return fb(n-1)+fb(n-2)

print(fb(6))
