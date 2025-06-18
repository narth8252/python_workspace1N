# 코딩테스트 연습>코딩테스트 입문>직각삼각형 출력하기
"""
"*"의 높이와 너비를 1이라고 했을 때, 
"*"을 이용해 직각 이등변 삼각형을 그리려고합니다. 
정수 n 이 주어지면 높이와 너비가 n 인 직각 이등변 삼각형을 출력하도록 

 입출력 예
입력 3
출력: 첫째줄에 * 1개, 둘째줄에 * 2개, 셋째줄에 * 3개출력
*
**
***
"""
# n = int(input())
# for i in range(1, n+1 ):
#     print('*'*i)


# def triangle(n):
#     for i in range(1, n+1):
#         print('*'*i)

# n = int(input("삼각형의 높이: "))
# triangle(n)

#삼각형을 우측정렬?
# def triangle(n):
#     for i in range(1, n + 1): 
#i가 1번째부터 n+1이전 1씩 증가하며 반복
#         # 앞쪽 공백(n - i)개 출력
#         print(' ' * (n - i) + '*' * i)

# n = int(input("삼각형의 높이(정수)를 입력하세요: "))
# triangle(n)

#삼각형을 트리처럼 가운데정렬?
def tree(n):
    for i in range(1, n + 1):
        # 앞쪽 공백 (n - i)개 출력
        print(' ' * (n - i) + '*' * (2 * i - 1))

n = int(input("삼각형의 높이(정수)를 입력하세요: "))
tree(n)
