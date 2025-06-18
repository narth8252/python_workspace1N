# mymodule2.py 문제풀이

def isEven(num):
    if num % 2 == 0:
        return True

def toUpper(s):
    # return s.upper() 이거쓰지말고 함만들어봐라
    #A -65, a-97 둘사이에 32차이, 소문자를 대문자로-32, 대문자를소문자로+32
    temp =""
    for c in s:
        # if ord('a') <= ord(c) <= ord('z'):  #소문자 범위체크
        if ord(c)>=ord('a') and ord(c)<=ord('z'): #소문자일때
            c = chr(ord(c)-32) #소문자 → 대문자 'a'-97 -32 =65 → 문자바꾼다
        temp = temp + c #문자추가: 다른문자 왔을땐 넘겨줘야되니까 +
    return temp

if __name__ == "__main__":
    print(isEven(4))  # True
    print(isEven(3))  # False
    print(toUpper('asterisk'))  # ASTERISK

# ord(c) : 문자를 유니코드 숫자로
# chr(n) : 유니코드 숫자를 다시 문자로
# 'a'는 97, 'A'는 65 → 소문자에서 32를 빼면 대문자
# if ord('a') <= ord(c) <= ord('z'): c가 소문자일 때만 변환
#유니코드 숫자는 문자(글자)를 숫자로 바꾼 값을 말합니다. 
# 컴퓨터는 모든 문자를 숫자로 저장하고 처리하니까요.