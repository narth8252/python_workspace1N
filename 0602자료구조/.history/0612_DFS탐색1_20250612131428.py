#0612 pm 1시  https://wikidocs.net/206317 
#다빈치코딩 알고리즘 02. 중급 알고리즘 06. 그래프알고리즘(DFS & BFS) 01. 인접 행렬의 DFS, BFS

#map은 list요소의 앞에서 전달해준 수식이나 함수 적용한다.
data = map(int,input().split()) #문장으로 받는다. 문장을slice해서 단어로 만들기
print(data)
"""
#위의 한줄문법 생각안나면 for루프 써도 됨. 노가다로.(시간안잡아먹음)
data = int,input().split()

n = int(data[0])
m = int(data[1])
print(n, m)
"""

"""
지도의 좌표를 인덱스로 표현한것과 비슷합니다. 
아래를 보면 마치 1이 벽이고 0이 길이라고 생각하면 됩니다. 
각각의 인덱스들은 상하좌우가 연결되어 있어 빈칸은 이동 가능하고 1은 이동할 수 없습니다. 그럼 0, 0 위치에서 가장 마지막 위치인 5, 5 위치로 이동하는 DFS 탐색을 만들어 보겠습니다.
먼저 입력을 받아야 합니다. 입력은 아래와 같은 형태로 들어옵니다.
6 6
0 1 1 1 1 1
0 1 0 0 0 1
0 1 0 1 0 1
0 1 0 1 0 0 
0 0 0 1 1 0
1 1 1 1 1 0
"""
N, M = map(int, input().split())
#print(N, M)

for i in range()