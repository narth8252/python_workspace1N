# 코딩테스트 연습>코딩테스트 입문>삼각형의 완성조건 (1)



"""
문자열 배열 strlist가 매개변수로 주어집니다. 
strlist 각 원소의 길이를 담은 배열을 return하도록 
solution 함수를 완성해주세요..

입출력 예
strlist	result
["We", "are", "the", "world!"]	[2, 3, 3, 6]
["I", "Love", "Programmers."]	[1, 4, 12]
["We", "are", "the", "world!"]의 각 원소의 길이인 [2, 3, 3, 6]을 return합니다.
"""
strlist = ["We", "are", "the", "world!"]
def solution(strlist):
    result = []

    for i in strlist:
        result.append(len(i))
    return result

"""
"""

#다른풀이
def solution(strlist):
    answer = list(map(len, strlist))
    #             map(function, iterable)함수: 주어진 iterable의 각요소에 func적용
    #             strlist의 각 문자열 길이를 순서대로 계산
    #        map함수의 결과타입은 map객체인데, list()로 감싸 변환
    return answer

#리스트 컴프리헨션(List Comprehension) 문법
def solution(strlist):
    return [len(str) for str in strlist]
            #                   strlist의 각요소(문자열)를 차례로 하나씩 꺼내 
            #            str변수에 저장
            # len(str)으로 문자열길이 구함
            #그 값을 새리스트[]에 차례대로 넣기

#반복(iteration)할 수 있는 객체
#for문이나 함수예:list(list(),tuple(),str(),dict,set, map(), sum()등)에서 차례로 꺼내 처리할수있는 객체
