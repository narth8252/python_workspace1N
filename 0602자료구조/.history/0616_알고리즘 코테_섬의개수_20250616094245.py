# https://wikidocs.net/206352
#다빈치코딩알고리즘6-1.인접행렬의DFS,BFS 01.섬의개수[백준4963]
# 지도상의 섬의 개수를 구하는 문제 입니다. 우리가 원하는 것은 연결된 섬이 몇개나 있는지 알고 싶은것이기 때문에 BFS가 아닌 간단한 DFS로 쉽게 구할 수 있습니다.
#코테에 자주나오는데 기본문제라 이거보다 어렵게 나옴.
# 다만 이 문제의 함정은 가로, 세로 또는 대각선 으로 갈 수 있다는 것입니다. 상하좌우 뿐만 아니라 대각선 방향까지 고려해서 문제를 해결해야 합니다.
"""

"""
#다빈치알고리즘 풀이는 할때마다 선만체크, 쌤풀이는 할때마다 값체크
#노가다로 일단 쓰자
M = 5 #열개수 너비
N = 4 #행개수 높이
arr = [
    [1, 0, 1, 0, 0], 
    [1, 0, 0, 0, 0], 
    [1, 0, 1, 0, 1], 
    [1, 0, 0, 1, 0], 
       ]

visited = [[0]*M for _ in range(N)]
# print(visited) #[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
def printArray(arr, N): #배열과 값
    for a in range(0, N):
        print(a)

printArray(visited)

#재귀호출 → 내부스택 사용
def dfs(y,x):
    #방문표시
    visited[y][x] = 1
    printArray(visited)
    #탐색이 가능한 좌표배열을 만든다
    #헷갈리면 써라 y상하좌우대각선
    dx = [0, 0, -1, 1, -1, -1, 1, 1]#좌우x
    dy = [-1, 1, 0, 0, 1, -1, 1, -1]#상하y
    print(y,x)

def solution(arr, M, N):
    for i in range(0, N): #(행,높이) 행이 먼저 나와야 편함
        for j in range(0, M): #열, 너비
            #방문안했고 섬이면 dfs호출하기
            if visited[i][j]==0 and arr[i][j]==1:
                dfs(i,j) #섬은0일거고 탐색한번할때마다 1개씩추가, dfs가 3번돔
                Island_cnt += 1

    return Island_cnt

solution(arr, M, N)


