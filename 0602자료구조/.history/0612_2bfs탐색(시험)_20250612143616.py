#0612 pm 2:20 외워야되는 코드임(시험70%) https://wikidocs.net/206317
#BFS탐색 : 큐를 사용한다.
#
"""데이터에 줄끝에 안보이는 공백도 없어야함.
6 6
011111
010001
010101
010100
000110
111110
"""
N, M = map(int, input().split())
arr = []
for i in range(N):
    #map함수는 iterable해서 아직 진행이 안되서  #list는 iterable아님
    #filter, range, zip 등등은 for문 안에서 호출하거나 list로 감싸줘야 작동함
    temp = list(input()) #str → list
    # print(temp)
    temp = list(map(int, temp))
    arr.append(temp)

def printArray(arr, N):
    for i in range(N):
        print(arr[i])

printArray(arr, N)

visited = [[False]*M for _ in range(N) ] # _ 언더바는 내보낼거 없이 돌리기만하겠다는 말 

from collections import de
def bfs(y, x): #시작값넣어라, (행y,열x)로 넣는게 편함
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    visited[y][x] = True #방문했었음
    #bfs는 Queue 활용한 탐색법

    q = deque()
    q.append((y,x)) #튜플로 추가하기
    while q: #큐가 빌때까지
        pass

