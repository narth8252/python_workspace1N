# 코딩테스트 연습>배열 회전시키기
# https://school.programmers.co.kr/learn/courses/30/lessons/120844
""" 
정수가 담긴 배열 numbers와 문자열 direction가 매개변수로 주어집니다. 
배열 numbers의 원소를 direction방향으로 한 칸씩 회전시킨 배열을 return

입출력 예
numbers	                    direction	result
[1, 2, 3]	                "right"  	[3, 1, 2]
[4, 455, 6, 4, -1, 45, 6]	"left"	    [455, 6, 4, -1, 45, 6, 4]
right이면 인덱스를 +1옮기고
left면 인덱스를 -1 옮겨 출력
회전로직:리스트슬라이싱
"""
# 리스트 안의 값 위치 바꾸기 (인덱스 위치 변경)
# 예: 리스트 [10, 20, 30, 40]에서 인덱스 0과 2의 값을 바꾸고 싶다면:
data = [10, 20, 30, 40]
data[0], data[2] = data[2], data[0]
print(data)  # [30, 20, 10, 40]

#풀이
def solution(numbers, direction):
    # answer = []
    if direction == "right":
        return numbers[-1] + numbers[:-1] #마지막원소부터 끝까지 
#data[:-1] 처음~마지막전:마지막요소를 앞에붙이고, 나머지 앞쪽요소들을 뒤에붙여서, 
# 결과적으로 오른쪽으로 한 칸 민 리스트를 만든다.
#data[-1:] 마지막전~끝:5 + data[:-1] 처음~마지막전: 1,2,3,4
    else: #direction == "left":
        return numbers[1:] + numbers[0]
    return answer
    
data = [1,2,3,4,5]
right = data[-1:] + data[:-1] #
print(right) #[5, 1, 2, 3, 4]

#오른쪽으로 n칸밀기
def rotate_right(data, n):
     n = n%len(data) #리스트길이 초과방지
     return data[-n:] + data[:-n]

#왼쪽으로 n칸밀기
def rotate_left(data, n):
     n = n%len(data)
     return data[n:] + data[:n]

#문자열회전(슬라이싱)
#오른쪽으로 n칸밀기
def rotate_str_right(s,n):
     n = n % len(s)
     return s[-n:] + s[:-n]


#슬라이싱은 파이썬만되니까, 1번 돌리는거 만들고 while루프로 n번돌리기
#1.왼쪽으로 1번회전하기
def Lrotate(arr):
    #힘드니까 왼쪽으로 돌리기
    # arr= [1,2,3,4,5]
    # arr= [2,3,4,5,1] #맨앞에 1이 맨뒤로 이동
    # temp = arr[0]
    # arr[0] = arr[1]
    # arr[1] = arr[2]
    # arr[2] = arr[3]
    # arr[3] = arr[4]
    # arr[4] = temp
    temp = arr[0]
    for i in range(1, len(arr)):
        arr[i-1] = arr[i]
    arr[-1] = temp

#3.오른쪽으로 1번회전하기
def Rrotate(arr):
    # arr= [1,2,3,4,5]
    # temp = arr[4]
    # arr[4] = arr[3]
    # arr[3] = arr[2]
    # arr[2] = arr[1]
    # arr[1] = arr[0]
    # arr[0] = temp
    
    temp = arr[len(arr)-1]
    for i in range(len(arr)-2, -1, -1):
    # for i in range(len(arr)-1, -1, -1):
        # arr[i] = arr[i-1] # arr[i-1]하면음수나오니까
        arr[i+1] = arr[i] # arr[i-1]하면음수나오니까
    arr[0] = temp  

#2.호출하기
def main(arr, n, direction):
    if direction=="L":
        for i in range(0, n):
            Lrotate(arr)
    
    else:
        for i in range(0, n):
            Rrotate(arr)

#4.테스트프린트
arr=[1,2,3,4,5,6,7,8,9,10]
main(arr, 5, "L") #10바퀴돌면 원상복귀
print(arr) #[6, 7, 8, 9, 10, 1, 2, 3, 4, 5]
main(arr, 5, "R") #10바퀴돌면 원상복귀
print(arr) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



"""
def solution(numbers, direction):
    cnt = 0
    while cnt < numbers: 
        if direction == 'right':
            return [numbers+1]
        elif direction == 'left':
            return [numbers-1]
        return [numbers]
          
def solution(n, direction, count=1):
    answer = []
    for _ in range(count):
        if direction == 'right':
            n+=1
        elif direction == 'left':
            n-=1
        answer.append()
    return answer

#deque
from collections import deque

def solution(numbers, direction):
    numbers = deque(numbers)
    if direction == 'righr':
        numbers.rotate(1)
    else:
        numbers.rotate(-1)
    return list(numbers)


#숫자돌리기
def rotate_righ(lst, n):
    count = 0
    while count < n:
        last = lst[-1]
        lst = [last] + lst[:-1]
        print(lst)  # 회전 결과 출력
        count += 1

#문자돌리기
def rotate_left(s, n):
    s = list(s)
    count = 0
    while count < n:
        first = s[0]
        s = s[1:] + [first]
        count += 1
    return ''.join(s)
"""