"""0604 pm3시 재귀호출
재귀호출의 핵심 내용
재귀호출이란, 함수가 자신을 다시 호출하는 방식입니다.
이는 트리 구조 탐색이나 특정 알고리즘(예: 퀵정렬, 분할 정복)에서 유용하게 쓰입니다.
의도치 않은 재귀 호출이 계속될 경우, 호출이 멈추지 않고 무한히 반복되어 **스택 오버플로우(Stack overflow)**가 발생할 수 있습니다.
이러한 문제를 방지하기 위해서는 재귀 호출을 중단하는 조건이나 제한 조건이 필요합니다.

callB()   callA()
의도치않은 재귀호출 안끝난다.
서로 계속 호출 반복하면 스택에 계속 쌓여서 스택overflow가 발생한다.
무한대로 호출하지 않게 만들어줘야한다

장점: 특별한경우 코드간결, 트리순회시 재귀호출 유용
     (ex.퀵정렬알고리즘,하노이탑알고리즘)
단점: 메모리를 많이 사용하고 속도느림.

def callA():
    if 조건:  # 재귀 종료 조건
        return
    # 재귀 호출
    callB()

def callB():
    if 조건:  # 재귀 종료 조건
        return
    # 재귀 호출
    callA()
"""
#이 예시는 바람직하지않지만 구조보려고 함
#반드시 언젠간 끝내야해서, 매개변수 갖고다니면서 이값을 증가 또는 감소시켜서
#끝나는 상황 만들어야함
#1~10까지 출력하려고 한다.
#1, 2, 3, 
# 함수인자n을 통해 재귀 호출 시 값을 계속 전달하며,
# n이 10보다 크면 종료(return), 그렇지 않으면 다시 recursive_call(n+1)을 호출하는 방식입니다.
def recursive_call(n): #내가 나를 호출
    if n > 10: #함수종료조건필수:안하면 무한루프
        return 
    recursive_call(n+1) #+1이든 -1이든 해서 계속호출해야함

recursive_call(1)
#1  recursive_call(2)
#1 2  recursive_call(3)
#1 2 3  recursive_call(4)
#1 2 3 4  recursive_call(5)
#..... #for문쓰지 싶겠지만 한번 해보는 것임
# 내부에 호출 순서가 있음을 이용해, 함수가 호출될 때마다 n 값이 변경되어, 재귀 호출이 끝나는 시점이 명확하다는 점입니다.
# 이렇게 하면 무한 재귀를 방지할 수 있고, 명시적 종료 조건이 있어 안전하게 사용할 수 있습니다.

#역으로 뒤집기
def recursive_call2(n):
    if n<0:
        return #끝내는 함수종료조건
    print(n) #위에 if문으로 함수종료조건없으면 무한루프
    recursive_call2(n-1)

recursive_call2(10)

#피보나치수열
"""
각각의 수가 바로 이전 두 수의 합으로 이루어진 수열입니다.
일반적으로 시작 값으로 0과 1을 사용하며, 이후 수는 다음과 같이 만들어집니다:
F(0)=0,F(1)=1
F(n)=F(n−1)+F(n−2)(n≥2)

f(1) = 1
f(2) =1
f(3) = f(1) + f(2) = 2
f(4) = f(2) + f(3) = 3
f(5) = f(3) + f(4) = 5
f(n) = f(n-2) + f(n-1) = 8
"""
# 재귀를 이용한 피보나치 함수 예제
def fibonacci(n):
    if n==1 or n==2:
        return 1
    return fibonacci(n-2) + fibonacci(n-1)
print("3번째요소", fibonacci(3))
print("6번째요소", fibonacci(6))

#1 ~ n까지의 합계 구하는 걸 재귀호출로 작성하기
# 1+2+3+....(n-1)= f(n)
# f(n) = n + f(n-1)
# f(1) = 1
# f(2) = f(1)+2
# f(3) = f(2)+3
# f(n) = f(n-1)+n
#꼬리를무는 재귀호출 예제
#for문이 빠름. 재귀호출은 메모리 엄청 사용. 이건 예시라 간단.
def mysum_recursive(n):
    if n<=1:
        return 1
    return mysum_recursive(n-1)+n
print(mysum_recursive(10))

#for문
def mysum_for(n):
    total = 0
    for i in range(1, n+1):
        total += i
    return total