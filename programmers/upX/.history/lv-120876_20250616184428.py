# 코딩테스트 연습>겹치는 선분의 길이


""" 
선분 3개가 평행하게 놓여 있습니다. 세 선분의 시작과 끝 좌표가 
[[start, end], [start, end], [start, end]] 형태로 들어있는 
2차원 배열 lines가 매개변수로 주어질 때, 
두 개 이상의 선분이 겹치는 부분의 길이를 
return 하도록 solution 함수를 완성해보세요.
lines가 [[0, 2], [-3, -1], [-2, 1]]일 때 그림으로 나타내면 다음과 같습니다.

선분이 두 개 이상 겹친 곳은 [-2, -1], [0, 1]로 길이 2만큼 겹쳐있습니다.

입출력 예
lines	                    result
[[0, 1], [2, 5], [3, 9]]	2
[[-1, 1], [1, 3], [3, 9]]	0
[[0, 5], [3, 9], [1, 10]]	8
두 번째, 세 번째 선분 [2, 5], [3, 9]가 [3, 5] 구간에 겹쳐있으므로 2를 return 합니다.
"""
def solution(lines):
    answer = 0
    # return answer
    
    temp = [[0]*200 for _ in range(3)]
    #for line in temp:
    #   print(line)
    
i=0
for start, end in lines:
    for j in range(start+100, end+100):
        temp[i][j] = 1
    i+=1
    
#for line in temp:
#   print(line)

cnt = 0
for j in range(0,200):
    if temp[0][j] == temp[1][j] == temp[2][j]==1: #셋다겹칠때
        cnt+=1
    else:
        #둘만겹칠때
        if temp[0][j] == temp[1][j] ==1:
            cnt+=1
        if temp[0][j] == temp[2][j] ==1:
            cnt+=1
        if temp[1][j] == temp[2][j] ==1:
            cnt+=1

#print(cnt)
return cnt





# def solution(age):
#     answer = ''
#     for ch in age:
#         answer += (ord(ch) - ord('a'))
#     return int(answer)

# def solution(age):
#     result = ''
#     for digit in str(age):
#         result += chr(ord('a') + int(digit))
#     return result