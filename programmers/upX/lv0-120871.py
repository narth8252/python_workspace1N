# 코딩테스트 연습>배열 회전시키기

""" 
3x 마을 사람들은 3을 저주의 숫자라고 생각하기 때문에 
3의 배수와 숫자 3을 사용하지 않습니다. 
3x 마을 사람들의 숫자는 다음과 같습니다.
"""
#마을숫자와 일반숫자를 같이 올림
def solution(n): 
    #answer = 0
    #n=목표숫자 <- normal숫자가 도달해야하는 목표숫자
    normal = 1
    town = 1

    while normal <= n:
        if town%3 == 0 or "3" in str(town): #숫자를 문자열로 바꿔 3제외
            town += 1
        else:
            normal += 1
            town   += 1
    print(normal)  #디버깅
    print(town -1) #디버깅

    return town-1

#마을숫자표 뽑아놓고 인덱스 가져오기
#정수 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#빼면 1 2   4 5   7 8   10 11       14    16 17    19 20
#의미 1 2   3 4   5 6    7  8        9    10 11    12 13
#23  "23" : str 3 in "n" == 숫자열에서 제외
def solution(n):
    answer = []
    #3들어가도 안됨. %3==0도 안됨
    for i in range(101): #100으로 하면 테스트출력100에서 오류
        if i % 3 != 0 and '3' not in str(i):
            answer.append(i)
    return answer[n-1] #출력해보니 숫자가 1씩 많이 나와서.