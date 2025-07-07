letter={'a','b','c','d'}#集合类型是无序的，所以每次匹配的结果都不同，输出结果也不同
number=[1,2,3,4]
for l,n in zip(letter,number):#同步迭代
    match l,n:
        case 'a',1:
            print('a1')
        case 'b',2:
            print('b2')
        case 'c',3:
            print('c3')
        case 'd',4:
            print('d4')