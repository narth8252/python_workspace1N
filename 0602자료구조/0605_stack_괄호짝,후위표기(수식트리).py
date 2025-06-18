# 좌충우돌,파이썬으로 자료구조04스택 04-02.괄호의 짝이 맞는지 확인
# https://wikidocs.net/192122
"""
괄호의 짝이 바르면 True, 바르지 않으면 False를 반환하는 함수를 작성하라.
예를 들어 ((a*(b+c))-d) / e는 괄호의 짝이 올바르지만, (((a*(b+c))-d) / e 는 괄호의 짝이 맞지 않는다. 괄호는 소괄호(())만 사용한다.

인터넷에서 스택 관련 문제를 검색할 때 나오는 문제 중 가장 낮은 난도의 문제다. 다른 방법도 있겠지만, 스택을 이용할 수 있다는 것을 안다면 의외로 쉽게 해결할 수 있다.

문제 분석하기
왼쪽 괄호가 나오면 가지고 있다가, 오른쪽 괄호가 나오면 가장 마지막에 나왔던 왼쪽 괄호와 짝이 맞는지 확인한다. 즉, 저장했던 가장 마지막 괄호부터 가져와서 확인하므로 스택을 이용한다.

수식에서 한 문자씩 가져와서 반복한다.
왼쪽 괄호가 나오면 스택에 넣는다.
오른쪽 괄호가 나오면 스택에서 왼쪽 괄호를 하나 뺀다.
스택이 비었으면, False 반환한다.
모두 검사했는데, 스택이 비었으면 True를 반환하고, 그렇지 않으면 False를 반환한다.
직접 구현한 Stack 클래스를 이용하여 푼다
"""
class MyStack:
    def __init__(self, size=10):
        if size < 10:
            self.size = 10   # 최소 크기를 10으로 하자.
        else:
            self.size = size

        self.stack=[]
        for i in range(0, self.size):
            self.stack.append(0)
        self.top = -1

    # push 함수
    # isFull 상태가 아니면 top 증가시키고 그 안에 값 넣기
    def isFull(self):
        if self.size-1 == self.top:
            return True
        return False
    
    def push(self, data):
        if self.isFull():
            return False
        self.top += 1
        self.stack[self.top] = data

    def print(self):
        i=0
        while i <= self.top:
            print(self.stack[i], end= " ")
            i += 1
        print()

    def isEmpty(self):
        if self.top == -1:
            return True
        return False
    
    def peek(self):
        if self.isEmpty():
            return False
        return self.stack[self.top]

    def pop(self):
        if self.isEmpty():
            return False
        item = self.stack[self.top]
        self.top -= 1
        return item

s = "((a*(b+c))-d) / e"
#한글자씩 읽어서 (와)만 필요
def isMatch(s):
  stack = MyStack(100)
  for c in s:
      if c=="(":
          stack.push(c)
      elif c==")":
          re = stack.pop()
            
  if stack.isEmpty() and re!=None:
      return True
  return False

"""250605 pm시
후위표기법 https://wikidocs.net/192124
 좌충우돌,파이썬으로 자료구조 04스택구현 4-03.후위표기법
정보처리기사에서 답 주는 문제임
후위 표기법의 수식을 계산하는 방법
위의 첫째 예제 (3+5*2)의 결괏값은 13이다. 후위 표기법으로 표시한 352/*+의 계산 방법을 나열해 보자.

피연산자(숫자)는 스택에 넣는다. 그럼, 스택의 상태는 [3, 5, 2]이다.
연산자를 만나면 스택에서 피연산자 두 개를 꺼내서 계산한다.
연산자 *를 만났으므로 스택에서 2와 5를 꺼내서 곱한다. 5 * 2 = 10
결괏값을 스택에 저장한다. 이제 스택의 상태는 [3, 10]다.
그다음에 연산자 +가 있으므로 스택에서 10과 3을 꺼내서 더한다. 답은 13이다.
이렇게 나열하고 보니, 중위 표기법처럼 괄호를 사용하지 않아도 연산의 우선순위가 명확하다는 장점이 있다. 
그래서 컴퓨터 입장에서는 후위 표기법이 효율적이다.
4 + 5 * 2 - 7/3
① 5 * 2
② 7 / 3
③ 4 +①
④ ③+②
수식트리
    -
  +   /
 4 * 7 3
  5 2
   순회방법                            표기법
  inorder  : LDR(전위:Root-Left-Right) infix      4 + 5 * 2 - 7 / 3
  preorder : DLR(중위:Left-Root-Right) prefix     - + 4 * 5 2 / 7 3
  postorder: LRD(후위:Left-Right-Root) postfix    4 5 2 * + 7 3 / -

  1.숫자면 스택에 push 4 5 2
  2.연산자면 2개를 pop left=5, right=2 연산수행해서 결과를 스택에 push
                      4 10
    + left=4 right=10  14
                     14 7 3
                     7/3=2.3334
                     14  2.3334
                     14 -2.3334= 11.6666
                     11.6666
"""
# Stack 클래스 이용하기
#  1.연산자의 우선순위를 비교해야 하므로, 사전을 이용하자.
#  2.문자열에서 한 글자씩 가져와서 반복한다.
#   -글자가 숫자이면 출력한다.
#   -숫자가 아니라 연산자이면
#     빈 스택이면 연산자를 추가한다.
#     빈 스택이 아니고, 스택의 마지막 연산자의 우선순위가 높으면 스택에서 pop하여 출력한다.
#     현재 연산자를 스택에 push한다.
# 3.스택에 남은 연산자가 있으면 모두 pop하여 출력한다.
#직접 구현한 스택을 이용한 함수

s = "4 5 2 * + 7 3 / -"
#한글자 받아가서 숫자인지 확인하는 함수
def isDigit(cd):
    if(ord(cd)>=ord('0') and ord(cd)<=ord('9')):
        return True
    return False

def isOperator(cd):
    if cd=="+" or cd=="-" or cd=="*" or cd=="/":
        return True
    return False

def operate(c, left, right):
    if c =="+":
        result = int(left) + int(right)
    elif c=="-":
        result = int(left) - int(right)
    elif c=="*":
        result = int(left) * int(right)
    else:
        result = int(left) / int(right)
    return result

def getResult(postfix): #수식을 받아가서 연산결과를 반환해야한다.
    stack = MyStack(100)
    for c in postfix:
        if isDigit(c):
            stack.push(c)
        elif isOperator(c):
            right = stack.pop()
            left = stack.pop()
            result = operate(c, left, right)
            stack.push(result)
    return stack.pop()

print(getResult(s))


"""
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item): self.items.append(item)
    def pop(self): return self.items.pop()
    def peek(self): return self.items[-1] if not self.is_empty() else None
    def is_empty(self): return len(self.items) == 0


def to_postfix(expression: str) -> str:
    op_precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    output = []
    stack = Stack()
    tokens = expression.replace(" ", "")

    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token in op_precedence:
            while (not stack.is_empty() and stack.peek() in op_precedence and
                   op_precedence[token] <= op_precedence[stack.peek()]):
                output.append(stack.pop())
            stack.push(token)
        elif token == "(":
            stack.push(token)
        elif token == ")":
            while not stack.is_empty() and stack.peek() != "(":
                output.append(stack.pop())
            stack.pop()  # "(" 제거

    while not stack.is_empty():
        output.append(stack.pop())

    return " ".join(output)

# #테스트 코드
# for expr in ("3 + 5 * 2", "3 * 5 + 2"):
#     print(f"{expr} -> {to_postfix(expr)}")

#리스트를 스택으로 사용하여 만들기
def to_postfix(expression: str) -> str:
    op: dict[str, int] = {"+":1, "-":1, "*":2, "/":2}
    res: str = ""
    s: list[str] = []
    for exp in expression:
        if exp.isnumeric():
            res += exp
        elif exp in op:
            if s and (op[exp] <= op[s[-1]]):
                res += s.pop()
            s.append(exp)
    while s:
        res += s.pop()
    return res

#테스트 코드
for expr in ("3 + 5 * 2", "3 * 5 + 2"):
    print(f"{expr} -> {to_postfix(expr)}")
 """
