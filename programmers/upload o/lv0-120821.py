# 코딩테스트 연습 > 코딩테스트 입문 > 배열 뒤집기


"""
정수가 들어 있는 배열 num_list가 매개변수로 주어집니다. 
num_list의 원소의 순서를 거꾸로 뒤집은 배열을 return하도록 
solution 함수를 완성해주세요.

제한사항
1 ≤ num_list의 길이 ≤ 1,000
0 ≤ num_list의 원소 ≤ 1,000

입출력 예
num_list	               result
[1, 2, 3, 4, 5]	        [5, 4, 3, 2, 1]
[1, 1, 1, 1, 1, 2]	    [2, 1, 1, 1, 1, 1]
[1, 0, 1, 1, 1, 3, 5]	[5, 3, 1, 1, 1, 0, 1]
"""
def solution(num_list):
    # answer = num_list[::-1] # 리스트를 뒤집는 슬라이싱 사용
    # return answer
# 리스트[시작인덱스 : 끝인덱스 : 간격]
# 시작인덱스: 슬라이싱을 시작할 위치 (생략 가능, 기본값은 0)
# 끝인덱스: 슬라이싱을 종료할 위치 (생략 가능, 기본값은 리스트 끝)
# 간격(step): 원소를 건너뛰는 크기 (생략 가능, 기본값은 1)
# 2. [::-1]의 의미
# :: 처음부터 끝까지 슬라이싱한다는 의미
# -1: "앞에서부터 뒤로", 즉 역방향으로 슬라이싱하라는 의미
    num_list.reverse()  # 리스트 자체를 뒤집기
    return num_list
# 파이썬 표준 라이브러리에는 reverse()라는 함수가 없습니다.
# 만약 리스트를 뒤집으려면:
# num_list[::-1] 방식으로 새 리스트를 만들어야 하고,
# 아니면 num_list.reverse()로 리스트를 직접 수정해야 합니다.


#다른풀이
def solution(num_list):
    return num_list[::-1]


def solution(num_list):
    num_list.reverse()
    return num_list

def solution(num_list):
    answer = []
    for i in range(len(num_list)-1, -1, -1):
        answer.append(num_list[i])
    return answer