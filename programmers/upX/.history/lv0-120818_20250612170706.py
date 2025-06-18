# 코딩테스트 연습>코딩테스트 입문>직각삼각형 출력하기
"""
"*"의 높이와 너비를 1이라고 했을 때, 
"*"을 이용해 직각 이등변 삼각형을 그리려고합니다. 
정수 n 이 주어지면 높이와 너비가 n 인 직각 이등변 삼각형을 출력하도록 

 입출력 예
입력 3
출력 첫째줄에 * 1개, 둘째줄에 * 2개, 셋째줄에 * 3개출력
*
**
***
"""
def solution(price):
    if price >= 500000:
        price *= 0.8
    elif price >= 300000:
        price *= 0.9
    elif price >= 100000:
        price *= 0.95
    return int(price)

# def solution(price):
#     if 100000 <= price:
#         return price - (price*0.05)
#     if 300000 <= price:
#         return price - (price*0.1)
#     if 500000 <= price:
#         return price - (price*0.2)



#다른풀이


