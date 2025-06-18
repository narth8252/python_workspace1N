# 코딩테스트 연습>코딩테스트 입문>옷가게 할인 받기

"""
머쓱이네 옷가게는 10만 원 이상 사면 5%, 30만 원 이상 사면 10%, 
50만 원 이상 사면 20%를 할인해줍니다.
구매한 옷의 가격 price가 주어질 때, 지불해야 할 금액을 return

 입출력 예
price	result
150,000	142,500
580,000	464,000
150,000원에서 5%를 할인한 142,500원을 return 합니다.
"""
def solution(s1, s2):
    count = 0
    for item in s1:
        if item in s2:
            count += 1
    return count

#다른풀이
def solution(s1, s2):
    return len(set(s1) & set(s2));

