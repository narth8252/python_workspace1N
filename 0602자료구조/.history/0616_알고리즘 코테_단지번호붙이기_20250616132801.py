# https://wikidocs.net/207175
#다빈치코딩알고리즘6-1.인접행렬의DFS,BFS 03.단지번호붙이기(정올 1996)[백준 2667]
#https://www.acmicpc.net/problem/2667
"""
예제 입력 1 
7
0110100
0110101
1110101
0000111
0100000
0111110
0111000

예제 출력 1 
3
7
8
9

1. 2차원배열 전체를 검색해야 한다. 어디가 섬인지 모르니까 
i, j 놓고 전체 배열을 검색한다. 
  첫번째 1을 찾는다. dfs나 bfs를 작동한다. 
  visited 배열         0, 0방이 1      
  0 0 0 0 0           1 0 0 0 0       1 0 0 0 0    
  0 0 0 0 0           0 0 0 0 0       1 0 0 0 0 
  0 0 0 0 0           0 0 0 0 0       0 0 0 0 0
  0 0 0 0 0           0 0 0 0 0       0 0 0 0 0



"""
#다빈치알고리즘 풀이는 할때마다 선만체크, 쌤풀이는 할때마다 값체크
#노가다로 일단 쓰자
M = 7 #열개수 너비
N = 7 #행개수 높이
arr = [
    [1, 0, 1, 0, 0], 
    [1, 0, 0, 0, 0], 
    [1, 0, 1, 0, 1], 
    [1, 0, 0, 1, 0]]

visited = [[0]*M for _ in range(N)]
# print(visited) #[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
def printArray(arr): #배열과 값
    for a in arr:
        print(a)

printArray(visited)

from collections import deque #큐라이브러리

#큐이용한 탐색
def bfs(y, x):
    dx = [0, 1, 1, 0]#좌우x
    dy = [0, 0, 1, 0]#상하y

    visited[y][x]=1 #방문
    q = deque()
    #1.큐에 現방문위치값 입력
    q.append((y,x))

    cnt = 0
    while q: #큐에 탐색할게 남아있는동안 탐색계속
        #큐에서 좌표값 하나 가져온다
        cy, cx = q.popleft() #큐에 넣을때 y값부터 넣었으므로 y부터 가져온다
        for i in range(len(dx)):
            ny = cy + dy[i] #새좌표 만든다
            nx = cx + dx[i]
            #벽인지 확인
            if cy<0 or ny>=N or nx<0 or nx>=M:
                continue
            if visited[ny][nx]==0 and arr[ny][nx]==1:
                #몇번만에 찾았는지 알아야하니, 이동한자릿수마다 cnt+1
                visited[ny][nx] = visited[cy][cx]+1
                q.append((ny, nx))
            return

bfs(0,0) #탐색하기
printArray(visited, N)

#재귀호출 → 내부스택 사용
def dfs(y,x):
    #방문표시
    visited[y][x] = 1
    # printArray(visited)
    #탐색이 가능한 좌표배열을 만든다
    #헷갈리면 써라 상하좌우대각선(둘이 일치하고 서로 어긋나게 만들면됨)안되면그림그려봐
    dx = [0, 0, -1, 1, -1, -1, 1, 1]#좌우x
    dy = [-1, 1, 0, 0, 1, -1, 1, -1]#상하y

    #8방향으로 새좌표 산출, 그래서 이동가능한지 확인
    for i in range(len(dx)):
        ny = y + dy[i] 
        nx = x + dx[i] 
        #이 좌표가 벽이면 다른좌표 확인(외워라)
        if ny<0 or ny>=N or nx<0 or nx>=M:
            continue #더이상 못가는 좌표는 버린다. 
                    #이 다음구문실행없이 다시 for문으로 돌아감
                    #방문안했거나 해당좌표1일경우엔 탐색지속
        if visited[ny][nx]==0 and arr[ny][nx]==1:
            dfs(ny, nx) #새좌표에서 또다른 탐색진행

        print(y,x)

def solution(arr, M, N):
    island_cnt=0 #섬의 개수 0 

    for i in range(0, N): #(행,높이) 행이 먼저 나와야 편함
        for j in range(0, M): #열, 너비
            #방문안했고 섬이면 dfs호출하기
            if visited[i][j]==0 and arr[i][j]==1: 
                dfs(i,j) #섬은0일거고 탐색한번할때마다 1개씩추가, dfs가 3번돔
                island_cnt += 1 

    return island_cnt

print(solution(arr, M, N))

"""
M, N = map(int, input().split()) 
arr = []
for i in range(N):
    temp = input().split()  # str-> list 
    #print(temp)
    temp = list(map(int, temp))
    arr.append(temp)

def printArray(arr, N):
    for i in range(N):
        print(arr[i])

printArray(arr, N)

visited = [ [False]*M for _ in range(N) ]

from collections import deque 
def bfs(y, x): #시작값 y:행, x:열
    dx = [-1,1,0,0,-1,-1,1,1 ]
    dy = [0,0,-1,1,-1,1,-1,1 ]
    visited[y][x] = True #방문을했음 
    #bfs는 큐를 활용할 탐색기법이다 

    q = deque()
    q.append((y,x)) #튜플로 추가하기 
    while q: #큐가 빌때까지 
        y, x = q.popleft()
        for i in range(len(dy)):
            #새로운좌표만들기; 
            ny = dy[i] + y 
            nx = dx[i] + x 
            if not(0<=nx<M and 0<=ny<N):
                continue 
            if arr[ny][nx]==0:
                continue 
            if not visited[ny][nx]:
                visited[ny][nx]=True 
                #큐에 방금 좌표값 추가 
                q.append((ny, nx))

def island_count():

    # cnt=0
    # bfs(0,0)
    # printArray(visited, N)
    # bfs(1,2)
    # printArray(visited, N)
    cnt=0 
    for i in range(0, N):
        for j in range(0, M):
            if arr[i][j] == 1 and not visited[i][j]:
                print(f"i={i}, j={j}")
                bfs(i,j)
                printArray(visited, N)
                print()
                print()
                cnt+=1 
    
    return cnt 

print (island_count())

#printArray(visited, N)        
"""
