# https://wikidocs.net/224878 
# 좌충우돌, 파이썬으로 자료구조 02-02. 회문(Palindromes) 찾기
# 펠린드롬(회문) 판별과정:문자열이 앞으로,뒤로 읽었을때가 같은지 확인
# 거울에 비친 자신의 모습과 똑같은지 한쌍씩 왼쪽오른쪽을 맞대어 확인하는 것.
# "이 왼손과 오른손이 일치하나요?"를 묻는 것처럼, 
# 문자열의 앞과 뒤가 같은지 차례차례 확인
"""펠린드롬 판별 흐름
1.문자열 길이 계산
  먼저, 검사하려는 문자열의 길이(ln)를 구합니다. (예: "level"은 길이 5)
2.중간까지만 비교
  문자열의 앞부분과 뒷부분을 비교하는데, 전체 길이의 절반까지만 반복.
  (홀수이면 중간 글자는 비교할 필요 없음)
3. 앞글자와 뒷글자 비교
 반복하는 동안,
 - 처음 문자(s[0])와 마지막 문자(s[ln-1])
 - 두 번째 문자(s[1])와 끝에서 두 번째(s[ln-2])
 이렇게 하나씩 비교합니다.
 만약 하나라도 다르면,
 "이 문자열은 펠린드롬이 아니다"라고 결론내리면 되죠.

4.모든 비교가 일치하면
  끝까지 비교했을 때 차이가 없다면,
  "이 문자열은 펠린드롬이다"라고 판단합니다.
"""

word = "racecar"
if word == word[::-1]:
    print(True)
else:
    print(False)
"""문제
절반을쪼개서 길이문자열의 길이 ln을 구한 후,
0부터 ln//2까지 반복하며,
s[i]와 s[ln - 1 - i]를 비교합니다.
이 방식은 문자열의 앞과 뒤를 비교하므로 효율적이고 직관적입니다.
길이를 ln len(문자열길이)-1
0        ln-0   arr[0] <-> arr[ln-0] #둘이 자리바꿈
1        ln-1   arr[1] <-> arr[ln-1] #둘이 자리바꿈
2        ln-2   arr[2] <-> arr[ln-2] #둘이 자리바꿈
3        ln-3   arr[3] <-> arr[ln-3] #둘이 자리바꿈
"""

# 250602 pm 1시 쌤풀이 알고리즘
import math
def palindrome(s):
    ln = len(s)-1
    #6    6/2 -3
    for i in range(0, math.ceil(ln/2)):
        if s[i] != s[ln-i]:
            return False
    return True #마지막까지 난ㅁ았다는 말은 회문(펠린드롬)이 성립

print( palindrome("madam"))
print( palindrome("tomato"))
print( palindrome("aabba"))

# def is_palindrome(s):
#     ln = len(s)  # 문자열 길이
#     for i in range(ln // 2):
#         # arr[i] 와 arr[ln - 1 - i] 비교
#         if s[i] != s[ln - 1 - i]:
#             return False  # 하나라도 다르면 회문 아님
#     return True  # 모두 일치하면 회문

# # 테스트 예시
# test_str = "level"
# if is_palindrome(test_str):
#     print(f"'{test_str}'은(는) 펠린드롬입니다.")
# else:
#     print(f"'{test_str}'은(는) 펠린드롬이 아닙니다.")