#점프투파이썬 05-4 예외처리 0513 1:45pm
#에러메시지 표기 try - except구문
print("---에러메세지표기 방법1.-----")
try:
    x = 10
    y = 0

    z = x/y
    print(f"x={x} y={y} z-={z}") #N/0513_에러.py
except ZeroDivisionError as e:  #division by zero(0으로 나눌수없다고 프로그램에 있음.)
    print(e) #에러메시지를 가져온다.

print("---에러메세지표기 방법2.-----")
#try-except-finally 기본 구조
#try:
    # 예외가 발생할 수 있는 코드
#except:
    # 예외 처리
#finally:
    # 예외 발생 여부와 상관없이 무조건 실행

try:
    x = int(input("정수 : "))
    y = int(input("정수 : "))
    z = x/y
    print(f"x={x} y={y} z-={z}") #N/0513_에러.py
except ZeroDivisionError as e:
    print("except: 0으로 나눌수없습니다.") #에러메시지를 가져온다.
finally:
    print("finally:는 에러나든말든 반드시 실행된다.")
    #finally: 자원 정리(예: 파일닫기, DB연결종료,네트워크 처리 등)에 많이사용
    #그래서 파일처리엔 with구문 쓰는것임.
    #파일,DB,네트워크연결...오류발생 close
