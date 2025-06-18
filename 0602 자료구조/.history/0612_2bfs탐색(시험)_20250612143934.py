#0612 pm 2:20 외워야되는 코드임(시험70%) https://wikidocs.net/206317
#BFS탐색 : 큐를 사용한다.
#
"""
이번에는 BFS 함수를 만들어 보겠습니다. 
DFS보다는 조금 어려운 느낌이 들 수 있지만 익숙해지면 쉽게 풀 수 있습니다. 
BFS는 그림과 같이 인접한 곳을 중심으로 원형으로 퍼져나가는 형태로 탐색이 진행 됩니다.
단계 0 (시작점)

         [S]

단계 1 (시작점 인접 노드 방문)

      [A]   [B]   [C]

단계 2 (1단계 인접 노드 방문)

   [D]  [E]  [F]  [G]

단계 3 (2단계 인접 노드 방문)

[H]  [I]  [J]  [K]  [L]

데이터에 줄끝에 안보이는 공백도 없어야함.
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

from collections import deque
def bfs(y, x): #시작값넣어라, (행y,열x)로 넣는게 편함
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    visited[y][x] = True #방문했었음
    #bfs는 Queue 활용한 탐색법

    q = deque()
    q.append((y,x)) #튜플로 추가하기
    while q: #큐가 빌때까지
        for i in range(len(dy)):
            #새로운 좌표 만들기(아까DFS랑 같은데 )
            ny = dy[i] + y
            nx = dx[i] + x

