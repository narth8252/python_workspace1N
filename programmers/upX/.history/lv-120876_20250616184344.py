# 코딩테스트 연습>겹치는 선분의 길이


""" 
선분 3개가 평행하게 놓여 있습니다. 세 선분의 시작과 끝 좌표가 
[[start, end], [start, end], [start, end]] 형태로 들어있는 
2차원 배열 lines가 매개변수로 주어질 때, 
두 개 이상의 선분이 겹치는 부분의 길이를 return 하도록 solution 함수를 완성해보세요.
lines가 [[0, 2], [-3, -1], [-2, 1]]일 때 그림으로 나타내면 다음과 같습니다.

line_2.png

선분이 두 개 이상 겹친 곳은 [-2, -1], [0, 1]로 길이 2만큼 겹쳐있습니다.
"""
# (chr(65)) #A, (chr(65+2)) #B,  (chr(65+3)) #C
age = 2345
answer = ""
for i in str(age):
    answer += chr(ord('A')+int(i)) #소문자로 바꾸고싶으면 a로 하면 됨.
print(ord('A'))







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