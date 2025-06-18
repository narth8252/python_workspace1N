#0609 pm1시 파이썬으로 트리구현 https://wikidocs.net/193702
#좌충우돌,파이썬으로 자료구조 07파이썬으로 트리 구현하기
""" 
preorder - node ->left - right

            G → D → E → B → F → C → A
inorder left - node - right
            A → B → D → 

🌳 이진(Binary)트리구조 개념도 (텍스트로 그린그림)
         A          #A는 root node     DLR: ABDECFG
       /   \                         INORDER: DBEAFCG
      B     C       #B,C는 A의자식노드  LRD: DEBFGCA
     / \   / \
    D   E  F  none  #D,E는 B의자식노드
   / \              #F는 C의자식노드
  G 
이 트리를 레벨 순서대로 배열에 저장하면 다음과 같다.
인덱스	0	1	2	3	4	5	6	    7
값  	A	B	C	D	E	F	None	G

  #07-01. 배열로 이진트리표현 https://wikidocs.net/193817
  0 1 2 3 4 5  6    7
  A B C D E F None  G
  A의 좌측자식 인덱스 1
  A의 우측자식 인덱스 2
  B의 좌측자식 인덱스 3
  B의 우측자식 인덱스 4
  C의 좌측자식 인덱스 5
  C의 우측자식 인덱스 6  

# 예시
 A(루트)의 인덱스는 0:
 - 왼쪽 자식 B의 인덱스: 2 × 0 + 1 = 1
 - 오른쪽 자식 C의 인덱스: 2 × 0 + 2 = 2
 B의 인덱스는 1:
 - 왼쪽 자식 D의 인덱스: 2 × 1 + 1 = 3
 - 오른쪽 자식 E의 인덱스: 2 × 1 + 2 = 4
 C의 인덱스는 2:
 - 왼쪽 자식 F의 인덱스: 2 × 2 + 1 = 5
 D의 인덱스는 3:
 - 왼쪽 자식 G의 인덱스: 2 × 3 + 1 = 7
인덱스 0은 비워두고, 인덱스 1부터 자료를 넣어 트리를 구성하기도 한다.
그렇게 하면 왼쪽 자식 노드의 인덱스는 (2 × n), 
오른쪽 자식 노드의 인덱스는 ( 2 × n + 1)이다. 
그럼 부모 노드의 인덱스는 자식 노드의 인덱스를 2로 나눈 몫만 취하면 된다.

자식 노드와 부모 노드
"""
class TreeNode:
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None

# 순회 함수들 (정상 동작 중)
def inorder(node):
    if node: #현재노드가 None이 아닐 때
        inorder(node.left)
        print(node.data, end="\t") #중간에 루트print
        inorder(node.right) 

def preorder(node):
    if node: #현재노드가 None이 아닐 때
        print(node.data, end="\t") #처음에 ptint
        preorder(node.left)
        preorder(node.right)

def postorder(node):
    if node: #현재노드가 None이 아닐 때
        postorder(node.left)
        postorder(node.right)
        print(node.data, end="\t") #마지막에 루트print

# 클래스(객체지향)를 사용하지않고 트리를 만들때 한계점 비교.
# 구체적으로는 insertNode() 함수로 새노드를 순서대로 자동삽입하려는 구조.
# 큐(Queue) 사용해서 레벨순서대로 빈자리를 찾아넣는 로직을 구현하려는 중.
def insertNode(data=None):
    #트리에 추가되기 위해서 노드 하나 만든다
    #트리가 비어있으면 새노드를 루트로 반환
    newNode = TreeNode(data)
    if root ==None:
        root = newNode
    # ABCDEFGHIJKLMN  <--------------- 
    #       A
    #     /   \
    #    B     C
    #   / \   / \
    #  D   E F   G
    # queue에 root 를 (parent입력하고)
    # queue에서 첫번째 root 가 나오면 root.left 가 None이면 추가 아니면 root.left를 큐에추가 
    # root.right 도 확인 None이면 여리가 붙이면 다시 root.right 를 큐에 넣는다          
    #내가 루트노드
    #내가 어디에 끼어들어갈지 

    # BFS로 빈 자리 탐색
#     q = deque()
#     q.append(root)

#     while q:
#         current = q.popleft()        
#         if current.left is None:
#             current.left = newNode
#             return root       
#         q.append(current.left)
        
#         if current.right is None:
#             current.right = newNode
#             return root        
#         q.append(current.right)

"""
큐를 이용한 레벨order - 너비우선탐색
1.큐를 초기화. 데큐써도됨
2.무조건 큐에 node넣고
3.큐가 비어있지않은동안 반복
    3-1 큐로부터 무조건 하나 가져온다
    3-2 데이터 출력
    3-3 가져온 노드의 left 가 None이 아니면 큐에 넣는다
    3-4 가져온 노드의 right가 None이 아니면 큐에 넣는다
"""
from collections import deque

def levelorder(node): #너비 탐색 
    queue = deque() #큐를 하나 만든다. 
    queue.appendleft( node ) #root 
    while len(queue)!=0: #while queue:큐가 empty될때까지
        current = queue.pop()  #current노드 하나 가져오기
        print(current.data, end="\t")
        if current.left != None: #if current.left:
            queue.appendleft( current.left ) #appendleft에 current.left 집어넣기
        if current.right !=None: #if current.right:
            queue.appendleft( current.right ) #appendleft에 current.right 집어넣기

#addNode함수:부모노드,원하는방;향,데이터주면, 그위치에 노드추가
root = None #list, dict 은 가능
def addNode(parent=None, left=True, data=None):
    #left값=True면 좌측, False면 우측에 붙이기
    global root #객체지향클래스안만들고 하려면 전역역변수global 써줘야함.
                #안쓰면 함수외부에서 변수에 접근불가

    temp = TreeNode(data)
    if parent == None:
        root = temp 
    else:
        if left:
            parent.left = temp 
        else:
            parent.right = temp 
    return temp 

def makeTree1():
    global root
    root = addNode(data="A") #우선노가다로 만들기 # root
    #parent값 안주면 디폴트로 None들어감
    levell = addNode(root, True, "B") #왼쪽에 B붙이기
    level2 = addNode(root, False, "C") #우측에 C붙이기
    addNode(level1, True, "D")  # B의 왼쪽에 D
    addNode(level1, False, "E") # B의 오른쪽에 E
    addNode(level2, True, "F")  # C의 왼쪽에 F
    addNode(level2, False, "G") # C의 오른쪽에 G

"""
    1.큐만들기 
    2.큐에 root 노드 추가 
        3. 추가될 위치 찾기 - 낑겨들어갈 위치찾기, 추가될때까지 
            3-1. 큐에서 하나 가져온다 
            3-2  가져온 노드의 left 가 None이면 여기에 추가하고 종료 
                None이 아니면 left값을 다시 큐에 넣는다 
            3-3  가져온 노드의 right 가 None이면 여기에 추가하고 종료 
                None이 아니면 right값을 다시 큐에 넣는다 
            
4. 모든 요소가 추가될때까지 1,2,3 을 반복한다. 

"""
#makeTree2("ABCDEFGHIK")
def makeTree2(simbol):
    global root 
    
    for c in simbol: 
        q = deque() #큐만들기 
        if root == None:
            root = TreeNode(c)
            continue  #한글자씩 가져와서 반복구조라서 range안써서 else안쓰고싶어서 continue
                    #continue아래문장 건너뛰고, 다시 조건식으로 이동
                    #대신 return못씀. 함수종료되니까.
        q.appendleft(root)
        #while루프 내가추가될때까지 
        #()언제 내가 추가되는지)애매하면 변수하나 만들어
        append = False 
        while not append:
            node = q.pop() #큐에서 하나가져와
            if node.left ==None:
                addNode(node, True, c)
                append = True #좌측에 붙고 끝나고 나감
            elif node.right ==None:
                addNode(node, False, c)
                append = True   #우측에 붙고 끝나고 나감
            else:
                q.appendleft(node.left)
                q.appendleft(node.right) 

#테스트코드
if __name__ == "__main__":
    # root = TreeNode("A")
    # root.left = TreeNode("B")
    # root.right = TreeNode("C")
    # root.left.left=TreeNode("D")
    # root.left.right=TreeNode("E")
    # root.right.left=TreeNode("F")
    # root.right.right=TreeNode("G")
    #makeTree("ABCDEFGHIJKL")
    #makeTree1()
    makeTree2("ABCDEFGHIK")
    print("inorder : ", end="\t")
    inorder(root)
    print() 

    print("preorder : ", end="\t")
    preorder(root)
    print() 
    
    print("postorder : ", end="\t")
    postorder(root)
    print() 
    
    print("levelorder : ", end="\t")
    levelorder(root)
    print() 
    
    #스택이나 큐, 시스템이 제공하는 큐를 쓰자 - 재귀호출 

# from collections  import deque 
# q = deque()
# q.appendleft('A')
# q.appendleft('B')
# q.appendleft('C')
# q.appendleft('D')

# print(len(q) )
# print(q.pop())
# print(len(q) )
# print(q.pop())
# print(len(q) )
# print(q.pop())
# print(len(q))

#           A
#         /   \
#        B     C
#       / \   / \
#      D   E F   G
# 출력 결과 (중위 순회: Left → Root → Right)
# inorder :       H       D       I       B       K       E       A       F       C       G
# preorder :      A       B       D       H       I       E       K       C       F       G
# postorder :     H       I       D       K       E       B       F       G       C       A
# levelorder :    A       B       C       D       E       F       G       H       I       K

