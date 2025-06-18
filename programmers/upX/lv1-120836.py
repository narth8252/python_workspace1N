# 코딩테스트 연습>코딩테스트 입문>순서쌍의 개수
"""
순서쌍이란 두 개의 숫자를 순서를 정하여 짝지어 나타낸 쌍으로 (a, b)로 표기.
자연수 n이 매개변수로 주어질 때 두 숫자의 곱이 n인 자연수 순서쌍의 개수를 return
 입출력 예
n	result
20	6
100	9
n이 20 이므로 곱이 20인 순서쌍은 (1, 20), (2, 10), (4, 5), (5, 4), (10, 2), (20, 1) 이므로 6
"""
def solution(n):
    answer = 0
    for a in range(n): #a는 0부터 n-1까지 하나씩 증가.
        if n % (a + 1) == 0: #(a+1)은 n의 약수
            answer +=1
    return answer


# a*b==n  b=n//a  1<=a<=b<=n
def solution(n):
    cnt = 0
    for a in range(1, n+1):
        if n % a == 0:
            b = n//a
            # if a <= b:
            cnt +=1
    return cnt        

print(solution(12))  # 출력: 3
print(solution(36))  # 출력: 5  → (1,36), (2,18), (3,12), (4,9), (6,6)
print(solution(17))  # 출력: 1  → (1,17)


#쌍 리스트 반환
def solution(n):
    pairs = []
    for a in range(1, int(n**0.5)+1):
        if n%a==0:
            b=n//aif a<=b:
            pairs.append((a,b))
    return pairs

print(solution(12))  # 출력: [(1, 12), (2, 6), (3, 4)]
print(solution(36))  # 출력: [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6)]
print(solution(17))  # 출력: [(1, 17)]