# 코딩테스트 연습>코딩테스트 입문>옷가게 할인 받기
"""
머쓱이네 옷가게는 10만원이상 사면5%, 30만원이상 사면 10%, 
50만원이상 사면 20%를 할인해줍니다.
구매한옷의 가격price가 주어질때, 지불해야할 금액을 return

 입출력 예
price	result
150,000	142,500
580,000	464,000
150,000원에서 5%를 할인한 142,500원을 return 합니다.
"""
def solution(price):
    if 100000 <= price:
        return price - (price*0.05)
    
    answer = 0
    return answer

#다른풀이


