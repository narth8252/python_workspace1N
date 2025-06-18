# 0617 am10:30
# 코딩테스트 연습>2025 프로그래머스 코드챌린지 2차 예선>완전범죄
# https://school.programmers.co.kr/learn/courses/30/lessons/389480
"""
완전탐색 - 모든경우의 수
완전탐색(Brute Force) + DFS(깊이 우선 탐색) 를 활용해서 모든 조합을 구하는 과정
숫자조합, 비밀번호 생성기, 순열/조합문제 등으로 확장가능
2개 조합 : 2중 for문
3개 조합 : 3중 for문
4개 조합 : 4중 for문
5개 조합 : 5중 for문
...무한정 확장불가해서 코드전체를 고쳐야한다
...이경우 재귀호출(어려움)사용
...for문은 횟수가 고정될 때만 쓸 수 있음 → 확장성 없음
...DFS(재귀)를 쓰면 어떤 길이든 확장 가능

리스트에 A 또는 B를 하나씩 쌓아가며 완성

끝까지 도달했을 때 (depth == n) 결과 출력

1. 개념 :A와 B 두 가지 선택지: n번 반복해서 만들 수 있는 모든 조합
        A B
2. (n=2)경우의수 실행 흐름 그리기 
0 0     A A
0 1     A B
1 0     B A
1 1     B B
dfs([], 0)
  ├── dfs(['A'], 1)
  │     ├── dfs(['A', 'A'], 2) → 출력: A A
  │     └── dfs(['A', 'B'], 2) → 출력: A B
  └── dfs(['B'], 1)
        ├── dfs(['B', 'A'], 2) → 출력: B A
        └── dfs(['B', 'B'], 2) → 출력: B B


3.(n=3)일 때 8가지의 경우의수 나옴
A A A  
A A B  
A B A  
A B B  
B A A  
B A B  
B B A  
B B B
dfs([], 0)
  ├── dfs(['A'], 1)
  │     ├── dfs(['A', 'A'], 2)
  │     │     ├── dfs(['A', 'A', 'A'], 3) → 출력
  │     │     └── dfs(['A', 'A', 'B'], 3) → 출력
  │     └── dfs(['A', 'B'], 2)
  │           ├── dfs(['A', 'B', 'A'], 3) → 출력
  │           └── dfs(['A', 'B', 'B'], 3) → 출력
  └── dfs(['B'], 1)
        ├── dfs(['B', 'A'], 2)
        │     ├── dfs(['B', 'A', 'A'], 3) → 출력
        │     └── dfs(['B', 'A', 'B'], 3) → 출력
        └── dfs(['B', 'B'], 2)
              ├── dfs(['B', 'B', 'A'], 3) → 출력
              └── dfs(['B', 'B', 'B'], 3) → 출력


4개는 16가지의 경우의 수 나옴
A A A A
A A A B
A A B A
A A B B
A B A A
A B A B
A B B A
A B B B
B A A A
B A A B
B A B A
B A B B
B B A A
B B A B
B B B A
B B B B

2. DFS를 이용한 전체 로직
DFS는 이렇게 작동해: 시작 → 스택에 넣고 깊게 들어감
                    끝에 도달하면 백트래킹(뒤로 빠지기)
                    그다음 경우로 다시 깊게 들어감
하는방법     DFS   DFS  DFS  DFS
A A A A     ['A', 'A', 'A', 'A'] -> 출력
스택에 쌓였다 없어지고 B를 출력
             DFS호출
A A A B     ['A', 'A', 'A', 'B'] -> 출력
             DFS  DFS호출
            ['A', 'A', 'B', 'A'] -> 출력
            DFS호출
            ['A', 'A', 'B', 'B'] -> 출력
            
템플릿 코드 (모든 n에 적용 가능)
def dfs(path, depth, n):
    if depth == n:
        print(path)
        return

    for ch in ['A', 'B']:
        dfs(path + [ch], depth + 1, n)

# 사용 예: 길이 3의 모든 A/B 조합
dfs([], 0, 3)

"""

