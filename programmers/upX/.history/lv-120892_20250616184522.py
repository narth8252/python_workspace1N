# 코딩테스트 연습>겹치는 선분의 길이

""" 
군 전략가 머쓱이는 전쟁 중 적군이 다음과 같은 암호 체계를 사용한다는 것을 알아냈습니다.

암호화된 문자열 cipher를 주고받습니다.
그 문자열에서 code의 배수 번째 글자만 진짜 암호입니다.
문자열 cipher와 정수 code가 매개변수로 주어질 때 해독된 암호 문자열을 returnreturn 합니다.
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