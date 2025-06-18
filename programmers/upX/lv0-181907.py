# 코딩테스트 연습>코딩테스트 입문>문자열의 앞의 n글자
"""
문자열 my_string과 정수n이 매개변수로 주어질 때, 
my_string의 앞의 n글자로 이루어진 문자열을 return 하는 solution 함수를 작성해 주세요.

 입출력 예
my_string	        n	result
"ProgrammerS123"	11	"ProgrammerS"
"He110W0r1d"	    5	"He110"
my_string에서 앞의 11글자는 "ProgrammerS"이므로 이 문자열을 return 
"""

def solution(my_string, n):
    answer = my_string[:n] #앞에서부터 n개 문자를 잘라서 반환
    return answer
"""
"""

#다른풀이