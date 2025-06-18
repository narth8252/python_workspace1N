#점프투파이썬05-4예외처리 > 여러개의 오류 처리하기
print("---에러메세지표기 방법4.-----")
#순서바꾸면안됨.
try:
    a = [1,2,3,4,5]
    b = a[5]  # 리스트의 인덱스 5에 접근
except ZeroDivisionError as e:
    print(e)
except IndexError as e:
    print(e)
except Exception as e:
    print(e)

# 리스트 a는 요소가 5개입니다: a[0] ~ a[4]까지.
# a[5]는 존재하지 않는 인덱스이므로 다음과 같은 예외가 발생합니다
#IndexError: list index out of range

#점프투파이썬05-4예외처리 > 오류일부러발생시키기.
#raise "예외문구" 강제예외발생
#원래하마수 종료구문은 return하는 일이 많다.
#return은 값도전송, 함수가 끝날때 마무리작업하고 나온다.
#return은 객체지향 이전부터 존재
#생성자에 오류가 발생했을때 어떻게 해? return 사용불가, 그래서 raise만들어놈
#raise ->정리작업도 하고 나온다.

class Test:
    def __init__(self): #심플한예제써서 에러확인
        # return True #여기서 return쓰면 에러남. 이미 빨간줄
        raise Exception("객체생성오류")
    
try:
    t1 = Test()
except Exception as e:
    print(e)

"""출력
list index out of range
객체생성오류
"""