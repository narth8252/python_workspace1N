# # 좌충우돌,파이썬으로 자료구조 04스택구현 04-03.후위표기법(Postfix no…
# https://wikidocs.net/192124
"""
문자열 수식을 받아서 후위표기법으로 바꾸는 함수를 작성하라.
(단, 피연산자는 10 미만의 수이다.)
 -입력: "3+5*2", 출력: "352*+"
 -입력: "3*5+2", 출력: "35*2+"
 
 #후위표기법(역폴란드 표기법, Reverse Polish Notation)
 연산자를 피연산자 뒤에 쓰는 연산표기법
  3+5*2 (중위표기법)
  352*+ (후위표기법) 결과값13

  1.피연산자(숫자)는 스택에 넣는다. 스택상태 [3, 5, 2]
  2. 연산자를 만나면 스택에서 피연산자 2개를 꺼내어 계산
  3.연산자 *를 만났으므로 2와 5를 꺼내 곱한다. 5*2=10
  4. 결과값을 스택에 저장. 스택상태[3, 10]
  5. 다음 연산자+가 있으므로 스택에서 10과3을 꺼내 +한다.답13

  중요한점은 연산자의 순서. 
  중위표기법의 연산자순서는 + * 이지만, 후위표기법은 * +이다.(우선순위기준)

  3+5*2
  문자 출력값 스택 설명
   3    3           피연산자는 출력
   +    3           빈 스택이므로 연산자를 스택에 push
   5    35    +     피연산자는 출력
   *    35    +*    *의 우선순위가 높으니,스택에push
   2    352   +*    피연산자는 출력
        352*+       스택에 있는것을 모두 pop

   3*5+2
   3    3           피연산자는 출력
   *    3     *     빈스택이므로 연산자를 스택에 push
   5    35    *     피연산자는 출력
   +    35*   *+    *의 우선순위가 높으니 pop하고 +를push
   2    35*2  +     피연산자는 출력
        35*2        스택에 있는것을 모두 pop

"""
#Stack class
# 문자열에서 한글자씩 가져와 반복
    # 글자가 숫자이면 출력
    # 연산자이면 - 빈스택이면 연산자추가
    #           - 빈스택 아니고, 스택의 마지막 연산자의 우선순위가 높으면 스택에서 pop하여 출력
    #           - 현재 연산자를 스택에 pop하여 출력

#함수
"""
def to_posfix(expression: str) -> str:
    op: dict[str, int] = {"+":1, "-":1, "*":2, "/":2}
    res: str =""
    s = Stack0602()
    for exp in expression:
    if exp.isnumeric():
        res += exp
    elif exp in op:
        if not s.is_empty() and (op[exp] <= op[s.peek()]):
            res += s.pop()
        s.push(exp)
    while not s.is_empty():
        res += s.pop()
    return res

#테스트
for expr in ("3 + 5 * 2", "3 * 5 + 2":):
    print(f"{expr} -> {to_postfix(expr)}")
""" 
"""
토큰화: 숫자, 연산자, 괄호로 분해
왼쪽부터 차례로 토큰 처리
숫자 -> 출력 큐에 바로 추가
왼쪽 괄호 ( -> 연산자 스택에 push
오른쪽 괄호 ) -> (를 만날 때까지 연산자 pop하여 출력 큐에 추가
연산자 -> 스택의 top 연산자와 우선순위 비교:
낮거나 같으면 pop하여 출력 큐에 추가
이후 현재 연산자 push
입력이 끝난 후, 스택에 남은 연산자를 출력 큐에 순서대로 pop
"""
from stack0602 import MyStack

#infix수식을 받아서 postfix 수식으로 전환하여 반환하는 함수
def isOperator(c):
    return c in "+=/*"

def isSpace(c):
    return c == " " #공백필수, 안하면 숫자끼리 붙어버림

#우선순위 가져오는 함수
def getPriority(c):
    if c in "+=":
        return 1
    elif c in "*/":
        return 2
    else:
        return 0

def postfix(expr):
    stack = MyStack(100)
    result=""
    for s in expr:
        #1.피연산자면 출력 456
        if s.isdigit():
            result += s
        elif isOperator(s): #연산자일 경우에 
            #and 연산은 1 and 1 => 1
            #0 and ? = 언제나0
            #while not stack.isEmpty()가 False이면 빠져나감.뒤 안하고
            #대부분언어가 and앞 수식이 False이면 뒤연산 안함
            #1 or ?(숏컷어쩌고)
            #                            peek()한애의 우선순위가 높으면 비교getPriority
            while not stack.isEmpty() and getPriority(stack.peek()) >= getPriority(s): #스택의 마지막꺼 빼와
                result = result + " " + stack.pop() #"공백"안넣어주면 숫자끼리 붙음
            stack.push(s)
        else: #공백일때
            result += s
    
    while not stack.isEmpty():
        result += " " + stack.pop() +" "
    return result

if __name__ == "__main__":
    str1 = "3+5*2"
    str2 = "3*5+2"
    answer1 = "352*+"
    answer2 = "35*2+"

    print(postfix(str1))
    print(postfix(str2))
    s = "4 5 2 * + 7 3 / -"
    #print

#0609 쌤파일.파이썬스택.py
from collections import deque

dq = deque() #데큐 라이브러리-양방향입출력 큐
dq.append('A')
dq.append('B')
dq.append('C')
dq.append('D')

print(dq.pop())
print(dq.pop())
print(dq.pop())
print(dq.pop())