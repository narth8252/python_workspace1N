#0610 pm2시
# 그리디 - 욕심쟁이, 탐욕
# 반드시 정렬을 해야 문제풀이 가능
#보통 스케줄 관련되고 정렬해서 최적화해야 풀수 있는 문제가 그리디 알고리즘임.
#젤 적은 수, 
"""
거스름돈으로 줄수있는 동전이 [500, 100, 50, 10원]일때,
거스름돈 금액 N원을 입력받아 동전의 최소개수를 구하라.
"""
#그리디1.
def get_change(n):
    #몫, 나머지 구해서
    coins = [500, 100, 50, 10]
    count = 0
    for coin in coins:
        count += n // coin
        n %= coin
    return count

print(get_change(1260)) #출력 6

#그리디2
"""
N개의 회의에 대해 각회의의 시작시간과 종료시간 주어진다
한회의실에서 사용할수있는 최대 회의 개수를 구하시오
(회의가 겹치면 안됨)즉, 종료시간 ≤ 다음 시작시간).

해결 방법 (그리디 알고리즘)
회의들을 종료시간 기준으로 오름차순 정렬합니다.
가장 빨리 끝나는 회의를 선택하여 회의실을 예약합니다.
이후 선택한 회의의 종료시간 이후에 시작하는 회의 중 가장 빨리 끝나는 회의를 선택하는 과정을 반복합니다.
meetings = [(1,4), (3,5), (0,6), (5,7), (8,9), (5,9)]
1,4
5,7
8,9
"""
def max_meetings(meetings):
                 #튜플에서2번째요소인 종료시간 기준으로 정렬
    meetings.sort(key=lambda x:x[1])

    count = 0 #최대사용 가능한 회의개수
    end_time = 0 #마지막 회의가 끝난 시간

    for start, end in meetings:
        if start >= end_time: #이전회의 종료시간 이후에 시작하는 경우
            count += 1
            end_time = end #현재 회의 종료시간으로 업데이트
    return count

#테스트
meetings = [(1,4), (3,5), (0,6), (5,7), (8,9), (5,9)]
print(max_meetings(meetings)) #3

#그리디3. 코딩테스트 연습> 코딩테스트 입문> 개미 군단
# https://school.programmers.co.kr/learn/courses/30/lessons/120837
""" 그리디 알고리즘
개미 군단이 사냥을 나가려고 합니다. 
개미군단은 사냥감의 체력에 딱 맞는 병력을 데리고 나가려고 합니다. 
장군개미는 5의 공격력을, 병정개미는 3의 공격력을 
일개미는 1의 공격력을 가지고 있습니다. 
예를 들어 체력 23의 여치를 사냥하려고 할 때, 
일개미 23마리를 데리고 가도 되지만, 
장군개미 네 마리와 병정개미 한 마리를 데리고 간다면 더 적은 병력으로 사냥할 수 있습니다. 
사냥감의 체력 hp가 매개변수로 주어질 때, 사냥감의 체력에 딱 맞게 최소한의 병력을 구성하려면 몇 마리의 개미가 필요한지를 return하도록 solution 함수를 완성해주세요.

 입출력 예
hp	result
23	5       (장군4+병정1=5)
24	6       (장군4+병정1+일1=5)
999	201     (장군199+병정1+일1=201)
hp가 23이므로, 장군개미 네마리와 병정개미 한마리로 사냥할 수 있습니다. 
따라서 5를 return합니다.

"""
#그리디3.
#0610 동전개수구하는 문제 푼 그리디임(최적화, 인원수 적게)
#몫, 나머지 구하고 하는거임

#for문
def solution(hp):
    ants = [5,3,1]
    answer = 0
    for ant in ants:
        answer += hp // ant
        hp %= ant
    return answer
print(solution(23))

#while문
def solution(hp):
    ants = [5,3,1]
    answer = 0
    i=0
    while i<3:
        mok = hp//ants[i]
        answer+=mok
        hp = hp%ants[i]
        i+=1
        
    answer += hp
    return answer
print(solution(23))

#0611 am11시
#그리디4.분할 가능한 배낭 문제 
"""
삼성 신입사원 알고리즘 문제
w = 50 #이 배낭은 50kg까지 담을수있다
items = [(60,10), (100,20),(120,30)] #(value,weight)

무게제한이 w인 배낭과 아이템n(예시3개)개가있다. 
각 아이템의 속성
value(가치) : 아이템을 배낭에 넣었을때 얻게되는 이익
weight(무게) : 아이템의 무게

이때, 각 아이템은 분할해서 넣을 수 있다.
가치를 최대로 담았을때 어느 정도 까지 담을 수 있는지?

-힌트: 가치/kg 당 구해서 집어넣고 해라
목적: 배낭에 담을 수 있는 아이템들의 조합 중 총 가치(value)의 합을 최대화하는 것
1.가치/무게 비율로 2.아이템 정렬 (내림차순:가치높은아이템부터 배낭에 넣기위해)
하나씩 넣되, 전체를 다 넣을 수 없으면 남은 용량만큼 비례해서 담는다.
누적 가치 합산 → 최종 정답

w = 50
 v/w         v가치  w무게   가치는 0부터 시작해서 하나씩 수행할때마다 +
                            무게는 배낭에서 빼야한다
 60/10 = 6    60     10     배낭에 다 담을 수 있을지 확인. w가 50이므로 배낭에담기
                            남은 무게는 40
100/20 = 5    160    20     남은 무게는 20                            
120/30 = 4    160 + 120*20/30 = 160+80 = 240
"""
def knapsack(w, items):
    #각 아이템에 대해(value, weight, w당v) 튜플생성
    items = [(v, w, v/w) for (v, w) in items]

    #kg당가치 기준 내림차순정렬(가치높은아이템부터 배낭에 넣기위해)
    items.sort(key=lambda x:x[2], reverse=True)
    #튜플에서2번째요소인 v/w기준정렬 원본변경.sort(,reverse내림차순)

    tl_value = 0 #전체가치 초기화
    remain_w = w #배낭의 남은용량

    for value, weigt, ratio in items:
        if weigt >= remain_w: 
            #아이템 전부 담기
            tl_value += value
            remain_w -= w
        else:
            #남은용량만큼 비례해서 담기
            tl_value += ratio*remain_w
            break

    return tl_value

#테스트
w = 50 #이 배낭은 50kg까지 담을수있다
items = [(120,30), (60,10), (100,20)] #(value,weight)
print(knapsack(w, items)) #최대가치 출력 240.0
#print(knapsack([(120,30), (60,10), (100,20)], 50))

#쌤풀이
def knapsack(items, w):
    items.sort(key=lambda x:x[1], reverse=True )
    value = 0

    #방법1. 노가다 if문 3번반복하며 누적
    if items[0][1] < w:
        value = items[0][0]
        w -= items[0][1] #잔량구함
    else: #kg당가치=가치/무게
        kg = items[0][0]/items[0][1]
        value += kg * w
        w -= w 
    #    i += 1

    if items[1][1] < w:
        value += items[1][0]
        w -= items[1][1]
    else: #kg당가치=가치/무게
        kg = items[1][0]/items[1][1]
        value += kg * w
        w -= w 

    if items[2][1] < w:
            value += items[2][0]
            w -= items[2][1]
    else: #kg당가치=가치/무게
        kg = items[2][0]/items[2][1]
        value += kg * w
        w -= w #w=0 무게초과이므로 종료

    #방법2. 3번반복하는걸 for문으로 변경
    for i in range(0, len(items)):
        if items[i][1] < w:
            value += items[i][0]
            w -= items[i][1]
        else: #kg당 가치(가치/무게)
            kg = items[i][0]/items[i][1]
            value += kg * w
            w -= w
            break #무게 초과

    return value

w = 50 #이 배낭은 50kg까지 담을수있다
items = [(120,30), (60,10), (100,20)]
print(knapsack(items,w)) #220.0
        