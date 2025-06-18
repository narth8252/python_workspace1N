# 코딩테스트 연습> 코딩테스트 입문> JadenCase 문자열 만들기


"""
JadenCase란 모든 단어의 첫 문자가 대문자이고, 그 외의 알파벳은 소문자인 문자열입니다. 단, 첫 문자가 알파벳이 아닐 때에는 이어지는 알파벳은 소문자로 쓰면 됩니다. (첫 번째 입출력 예 참고)
문자열 s가 주어졌을 때, s를 JadenCase로 바꾼 문자열을 리턴하는 함수, solution을 완성해주세요.
 입출력 예
s	return
"3people unFollowed me"	"3people Unfollowed Me"
"for the last week"	"For The Last Week"
"""

def solution(s):
    s = s.lower() #
    answer = ''

    #이중for문 안쓰고
    for i in range(0, len(s)):
        if s[i]>-"0" and s[i]<= "9": #숫자일경우 그냥 붙인다.
            answer += s[i]
        elif s[i]==" ":
            answer += s[i]
        else:
            #문장 첫글자였거나 자기앞글자가 공백이면 대문자로 바꿔서
            if i==0 or i>0 and s[i-1]==" ":
                answer += s[i].upper()
            else:
                answer += s[i]
    return answer

print( solution("for the last week"))