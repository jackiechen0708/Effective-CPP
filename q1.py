n = int(input())

def q1(lst):
    result = [lst[0]]
    for i in range(1,len(lst)):
        if result[-1][1] >= lst[i][0]:
            if result[-1][1] > lst[i][1]:
                result[-1] = lst[i]
        else:
            result.append(lst[i])

    return len(result)




for i in range(n):
    line = [int(j) for j in input().split()]
    intervals = [line[i:i+2] for i in range(0,len(line),2)]
    intervals.sort()
    print (q1(intervals))
