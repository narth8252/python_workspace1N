#0612 pm 2:20
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
    temp = list(input).append(list())
    temp = map(int, input().split())

    # arr.append( list(map(int, input().split())) )
