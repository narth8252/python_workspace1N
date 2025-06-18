#0612 pm 1시  https://wikidocs.net/206317 
#다빈치코딩 알고리즘 02. 중급 알고리즘 06. 그래프알고리즘(DFS & BFS) 01. 인접 행렬의 DFS, BFS
#코테랑 시험보려면 외워야함.

#map은 list요소의 앞에서 전달해준 수식이나 함수 적용한다.
# data = map(int,input().split()) #문장으로 받는다. 문장을slice해서 단어로 만들기
# print(data)
"""
map은 list 요소 앞
#위의 한줄문법 생각안나면 for루프 써도 됨. 노가다로.(시간안잡아먹음)
data = int,input().split()

n = int(data[0])
m = int(data[1])
print(n, m)
"""

""" 문제
01. 인접 행렬의 DFS, BFS
인접 행렬은 우리의 지도를 그래프 형태로 바꿔놓은 것으로 생각하면 쉽습니다. 
집에서부터 학교까지의 길을 지도로 그려보았습니다.
그림지도를 숫자로 단순화해보겠습니다.
지도의 좌표를 인덱스로 표현한것과 비슷합니다. 
1이 벽이고 0이 길이라고 생각하면 됩니다.
	0	1	2	3	4	5
0		1	1	1	1	1
1		1				1
2		1		1		1
3		1		1		
4				1	1	
5	1	1	1	1	1	
각각의 인덱스들은 상하좌우가 연결되어 있어 빈칸은 이동 가능하고 1은 이동할 수 없습니다. 
그럼 0, 0 위치에서 가장 마지막 위치인 5, 5 위치로 이동하는 DFS 탐색을 만들어 보겠습니다.
먼저 입력을 받아야 합니다. 입력은 아래와 같은 형태로 들어옵니다.

str → list → map(int, ) → list
6 6
0 1 1 1 1 1  
0 1 0 0 0 1
0 1 0 1 0 1
0 1 0 1 0 0 
0 0 0 1 1 0
1 1 1 1 1 0

6 6
011111  
010001
010101
010100 
000110
111110
"""

# N, M = map(int, input().split())
# #print(N, M)
# arr = []
# for i in range(N):
#     #map함수는 iterable해서 아직 진행이 안되서
#     #filter, range, zip 등등은 for문 안에서 호출하거나 list로 감싸줘야 작동함
#     #list는 iterable아님
#     temp = map(int, input().split())
#     arr.append(list(temp))
#     # arr.append( list(map(int, input().split())) )

#위것도 되는데 아래처럼 해도 됨.데이터 매번 넣기 힘드니까
N, M = (6,6)
arr = [
    [0, 1, 1, 1, 1, 1],  
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0], 
    [0, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0]]

#프로그램 짤때 출력함수 하나 만들어서 디버깅용으로 쓰면 알고리즘 개발시 편함.
#print(arr) 로 하면 옆으로 쭉 붙어있어서 보기불편함.
def printArray(arr, N):
    for i in range(0, N):
        print( arr[i] )
printArray(arr, N)


"""
DFS 함수 만들기 :인접행렬의 DFS 탐색
      0   1   2   3   4
     ------------------
 0 ㅣ 0                   둘러싸고 있는공간, 상하좌우 4개방향, 대각선4개방향
 1 ㅣ 0                   1, 0  0,0  2,0
 2 ㅣ 0                   방문테이블 만들고 내가 갔던 위치를 1로 바꿔서 다 표시한다.
"""
#방문테이블 파이썬 미리 메모리를 할당해서 1차원 []*개수
#for루프 안돌리고 컴프리헨션써서 1문장으로 축약
visited = [[False] * M for _ in range(N)]
#for루프로는?
visited=[] #Array여기에 넣어놓고 노가다로 시작하기도 함.
for i in range(N):
    visited.append([False]*M)
printArray(visited, N)


#깊이우선 탐색
def dfs(y, x):
    #매개변수로 전달받는 좌표는 현재 위치값
    visited[y][x] = True #방문했음

    #現좌표로부터 4방향 내지는 8방향 확인가능
    dx = [-1,1,0,0] #좌우  [-1,1, 0,0, -1,-1, 1,1]
    dy = [0,0,-1,1]
     #상하  변이-이동값

    for i in range()
    #새 좌표점 찾기
    ny = dy[0]+y
    nx = dx[0]+x
    #새좌표가 벽일수도 있고 이미 방문했던 곳일수도 있다
    if 0<= nx <N and 0<=ny <M: #벽임
        pass
    if arr[ny][nx]==1:
        pass
    if not visited[ny][nx]: #아직 미방문이면
        dfs(ny,nx) #새로운 출발점으로 해서 재귀호출
