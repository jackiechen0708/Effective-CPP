n = int(input())


def q3(lst):
    result = [lst[0]]
    for i in range(1,len(lst)):
        if result[-1][1] >= lst[i][0]:
            if result[-1][1] < lst[i][1]:
                result[-1][1] = lst[i][1]
        else:
            result.append(lst[i])

    return max([a[1]-a[0] for a in result])



for i in range(n):
    line = [int(j) for j in input().split()]
    intervals = [line[i:i+2] for i in range(0,len(line),2)]
    intervals.sort()
    print (q3(intervals))