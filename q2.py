n = int(input())


def q2(lst):
    add = [i[0] for i in lst]
    dec = [i[1] for i in lst]
    add.sort()
    dec.sort()
    result = 1
    i = 1
    j = 0
    cur = 1
    while i < len(add) and j < len(dec):
       if add[i] <= dec[j]:
           cur += 1
           result = max(result,cur)
           i += 1
       else:
           cur -= 1
           j += 1
    return result




for i in range(n):
    line = [int(j) for j in input().split()]
    intervals = [line[i:i+2] for i in range(0,len(line),2)]
    intervals.sort()
    print (q2(intervals))
