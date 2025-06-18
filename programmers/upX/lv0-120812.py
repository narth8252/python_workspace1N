# 코딩테스트 연습>코딩테스트 입문>최빈값 구하기
"""
최빈값은 주어진 값 중에서 가장 자주 나오는 값을 의미합니다. 
정수 배열 array가 매개변수로 주어질 때, 
최빈값을 return 하도록 solution 함수를 완성해보세요.
 최빈값이 여러 개면 -1을 return 합니다.

 제한사항
0 < array의 길이 < 100
0 ≤ array의 원소 < 1000

 입출력 예
array	            result
[1, 2, 3, 3, 3, 4]	3
[1, 1, 2, 2]	    -1
[1]                	1
[1, 2, 3, 3, 3, 4]에서 1은 1개 2는 1개 3은 3개 4는 1개로 최빈값은 3입니다.
"""
def solution(array):
    mydic = dict()
    
    #2. for i in range0,len(array0,len(array): 반복문
    for i in range(0, len(array)): #인덱스개수를len으로 셈, 0번방부터 넣어라.
        # 3.숫자를 딕셔너리에 저장 및 개수 세기
        if array[i] in mydic.keys(): #mydic:도 됨. value값만 줘도 됨
            mydic[array[i]] += 1
            #array[i]는 배열에서 i번째요소를 의미.
            #현재숫자가 딕셔너리에 있으면, 그숫자의개수를 1증가
        else:
            mydic[array[i]] = 1
            #처음 나온숫자면, 딕셔너리에 새로추가하고 개수를 1로 설정
            # 1.1 2.1 3.3 4.1
    
    #4.최빈값개수찾기
    max_val = -1 #초기화:빈도수 #최종값 3
    max_key = -1 #초기화:키값(숫자 자체)
    for key in mydic.keys():
        if  mydic[key] > max_val: 
            mydic[key] = max_val 

    cnt = 0
    max_key = 0
    for key in mydic.keys():
        if mydic[key] == max_val:
            max_key = key
            cnt += 1

    if cnt != 1:
        return -1
    else:
        return max_key


#다른풀이
def solution(array):
    freq = {}
    for num in array:
        freq[num] = freq.get(num, 0) + 1

    max_freq = max(freq.values())
    modes = [k for k, v in freq.items() if v == max_freq]

    return -1 if len(modes) > 1 else modes[0]

#다른풀이
def solution(array):
    temp = 0
    same = 0
    cnt = 0
    for i in set(array):
        cnt = array.count(i)
        if cnt > temp:
            temp = cnt
            answer = i
            same = 1

        elif cnt == temp:
            same += 1
    if same > 1:
        return -1
    else:
        return answer
    
""" 
def solution(array):
    # 1. 최빈값 후보와 등장 횟수 저장용
    max_count = 0
    mode = -1

    for i in array:
        count = 0
        for j in array:
            if i == j:
                count += 1

        if count > max_count:
            max_count = count
            mode = i
        elif count == max_count and i != mode:
            mode = -1  # 최빈값 여러 개일 때 -1로 고정

    return mode
"""


"""

"""
""" 
#다른풀이
import statistics
def solution(array):
    return statistics.median(array)
#결과값3개중2개만 맞음

#짝수개 숫자가 있을땐, 가운데 두숫자 더해서 나누기2 =중앙값
def solution(array):
    sorted_array = sorted(array) #배열정렬
    length = len(sorted_array)   #배열길이확인
    center = length //2         #중앙인덱스 계산
    
    if length %2==1: #홀수길이
        return sorted_array[center]
    else: #짝수길이 → 가운데 두 수의 평균
        return (sorted_array[center-1] + sorted_array[center])//2
        #1. center=length//2 중앙값
        #2. center-1 → 가운데 왼쪽 숫자 (index 1)
        #3. center → 가운데 오른쪽 숫자 (index 2)
        #4. 두값을 더해서 2로 나눔 → 평균값(중앙값)

    # median = sorted_array[center] #중앙값 추출
    # return median #반환
    # 결과값이 다르게 3개중 1개가 음수로 나옴

from collections import Counter

def solution(array):
    counter = Counter(array)                     # 1. 각 숫자의 빈도수 계산
    max_count = max(counter.values())            # 2. 가장 높은 빈도수 찾기
    mode = [num for num, cnt in counter.items() if cnt == max_count]  # 3. 최빈값 후보들
    
    if len(mode) == 1:
        return mode[0]                           # 4. 하나면 그 수
    else:
        return -1                                # 5. 여러 개면 -1

"""
def solution2(array):
    while len(array) != 0:
        for i, a in enumerate(set(array)): #set은 중복값 제거
            print("i=", i, "a=", a)
            array.remove(a)
        if i == 0: return a
    return -1

print(solution2([1,2,3,3,3,4,4,4,4]))