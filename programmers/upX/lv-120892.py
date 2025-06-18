# 코딩테스트 연습>겹치는 선분의 길이

""" 
군 전략가 머쓱이는 전쟁 중 적군이 다음과 같은 암호 체계를 사용한다는 것을 알아냈습니다.
암호화된 문자열 cipher를 주고받습니다.
그 문자열에서 code의 배수 번째 글자만 진짜 암호입니다.
문자열 cipher와 정수 code가 매개변수로 주어질 때 해독된 암호 문자열을 return

cipher                  	code	result
"dfjardstddetckdaccccdegk"	4	"attack"
"pfqallllabwaoclk"	        2	"fallback"
"""
def solution(cipher, code):
    answer = ''
    for i in range(code-1, len(cipher), code):
        answer += cipher[i]
    return answer

