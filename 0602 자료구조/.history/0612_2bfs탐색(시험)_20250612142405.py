#0612 pm 2:20 외워야되는 코드임(시험70%)
"""
6 6
011111  
010001
010101
010100 
000110
111110
"""
N, M = map(int, input().split())
#print(N, M)
arr = []
for i in range(N):
    #map함수는 iterable해서 아직 진행이 안되서
    #filter, range, zip 등등은 for문 안에서 호출하거나 list로 감싸줘야 작동함
    #list는 iterable아님
    temp = list(input()) #str -> list
    temp = list(map(int, temp))
    arr.append(temp)

def printArray(arr, N):
    for i in range(N):
        print(arr[i])

pr
