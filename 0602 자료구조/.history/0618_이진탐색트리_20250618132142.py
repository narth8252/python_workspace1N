#0611 pm2:25 좌충우돌,파이썬으로자료구조 09이진 탐색 트리
#https://wikidocs.net/195269
#이분검색 - 배열
#이진탐색트리
#데이터를 넣어서 트리를 만들때
#순서를 지키면서 만든다.
#이진트리 -> left, right 두개의 edge를 넣는다.
# 16  8  9  2  4  12 17  21  23  나보다 작은값은 왼쪽으로, 나보다 큰값은 오른쪽으로
#          16
#        8           17  
#     2     9            21
#   4          12            23


#Dict써도되고, 특정데이터타입만 저장됨
#책, 전화번호 등 만들수 있어서 클래스로 만들고 내부구조만 변경가능
class Data:
    def __init__(self, num):
        self.num = num

#이진트리 구축하기
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None #None안줘도 None으로 들어감
        self.right = None #None안줘도 None으로 들어감

#이진트리 구축하기
class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, data):
        #1.root노드가 있는지확인 root==None이 없으면 root노드를 만들기
        if not self.root:
            self.root = TreeNode(data)
            return #끝났음. 바로 함수종료