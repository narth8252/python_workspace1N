# mymodule2.py 
#0513 모듈설명후 패키지 실습 3-1
#신규폴더 mypackage1만들고 그 아래 방금만든 모듈.py 파일2개 넣어놓고 실행하면 가져와서 출력됨
#이거useModule.py는 workspace1N폴더에 있어야하고 나머지 __init__.py이랑 복붙한mymodule2.py랑 는 새로만든 mypackage1폴더에 있어야함.


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