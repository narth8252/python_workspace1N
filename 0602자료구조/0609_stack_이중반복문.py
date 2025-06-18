# 0609 pm 5시
# 좌충우돌,파이썬자료구조 04스택 04-05.자신보다 큰원소찾기
# https://wikidocs.net/192423
"""
문제
음이 아닌 정수 배열이 주어졌을 때, 각 원소의 오른쪽에 있는 원소 중에서 현재 원소보다 큰 값을 출력하되, 가장 근접한 원소를 출력하라. 현재 원소보다 큰 값이 없으면 -1을 출력하라.

예시 1
입력: [4, 5, 2, 25]
출력:
4 --> 5
5 --> 25
2 --> 25
25 --> -1

예시 2
입력: [13, 7, 6, 12]
출력:
13 --> -1
7 --> 12
6 --> 12
12 --> -1

가장 쉬우면서, 먼저 생각나는 방법은 이중 반복문을 사용하는 것이다. 
그런데, 이것을 스택으로 풀면 단일 반복문으로 풀 수 있다고 한다. 
일단 이중 반복문으로 풀어 본 후에 스택을 어떻게 활용할 수 있는지 고민해 보자.

힌트. 이중 반복문으로 풀기
인덱스 i는 0부터 n-1까지 반복
nge(다음으로 큰 원소)를 -1로 초기화
인덱스 j는 i+1부터 n-1까지 반복
인덱스 j의 원소가 인덱스 i의 원소보다 크면
nge에 인덱스 j의 원소를 대입
반복문을 벗어난다.
현재 원소와 nge를 출력한다
"""

def find_nge(arr: list[int]) -> list[int]:
    n: int = len(arr)
    for i in range(n):
        nge: int = -1
        for j in range(i+1, n):
            if arr[j] > arr[i]:
                nge = arr[j]
                break
        print(f"{arr[i]} --> {nge}")


#테스트 코드
find_nge([4, 5, 2, 25])

# 스택을 사용하여 풀기
def find_nge(arr: list[int]) -> list[int]:
    n: int = len(arr)
    s: list[int] = []
    res: list[int] = [-1] * n
    for i in range(n-1, -1, -1):
        while s:
            if s[-1] > arr[i]:
                res[i] = s[-1]
                break
            else:
                s.pop()
        s.append(arr[i])
    for i in range(n):
        print(f"{arr[i]} --> {res[i]}")


#테스트 코드
find_nge([4, 5, 2, 25])
