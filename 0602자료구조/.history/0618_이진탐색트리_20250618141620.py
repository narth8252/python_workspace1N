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
        print("****")
        self.root = None

    def insert(self, data=None):
        #1.root노드가 있는지확인 root==None이 없으면 root노드를 만들기
        if not self.root:
            self.root = TreeNode(data)
            return #끝났음. 바로 함수종료
        
        #1.내 노드가 들어갈위치찾기
        parent = None
        current = self.root #추척해들어가다보면 None인곳에 추가가능

        while current: #current가 None이 아닌동안
            if current.data.num == data.num:
                return #데이터중복되면 안넣을것임(중복데이터 배제)
            parent = current #現위치값저장. 나중에 parent와 연결
            #나보다 작은값은 왼쪽으로, 나보다 큰값은 오른쪽으로
            if data.num < current.data.num:
                current = current.left
            else:
                current = current.right
        #while문역할:터미널 노드는 edge가 없다. current값이 None일때까지 좌우움직이며 찾아감
        #그래서 뒤에 parent가 따라가며 위치값을 가져가야함
        # 커런트가 나고, 나는 패런트에 추가되야되는것임

        #노드만들어서 parent에 연결
        newNode = TreeNode(data)
        if data.num < parent.data.num:
            parent.left = newNode
        else:
            parent.right = newNode

    def inorder(self, node):
        if node==None:
            return
        self.inorder(node.left)
        print(node.data.num)
        self.inorder(node.right)

#데이터검색함수(사전,전화번호부)
    def search(self, key):
        current = self.root #검색시작위치
        count = 0 #찾은회수
        while current:
            if key.num == current.data.num: #찾았으니 함수종료
                return count
            
            count+=1 #못찾았을경우
            if key.num < current.data.num:
                current = current.left
            else:
                current = current.right

#데이터 삭제함수(난이도上)


if __name__=="__main__":
    bst = BinarySearchTree()
    arr = [16, 8, 9, 2, 4, 12, 17, 21, 23]
    for i in arr:
        bst.insert(Data(i))
    bst.inorder(bst.root)

#출력
print(bst.search(Data(16)))
print(bst.search(Data(2)))
print(bst.search(Data(26)))
