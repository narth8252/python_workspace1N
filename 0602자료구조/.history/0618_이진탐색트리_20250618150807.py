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
        
    #삭제함수안에 넣다가 코드가 기니까 find함수 만듬:이런기법은 파이썬에서만 가능
    def find(self, key):
        parent = None       #삭제될 노드의 부모와 자식연결
        current = self.root #삭제될 노드의 위치찾기
        find = False #못찾음
        while current and not find: #search랑 똑같지만 parent를 ??하므로 똑같이 씀. 튜플로 묶어도됨.
            if current.data.num == key.num:
                find = True #찾아서 따라나옴
            else: #없으면 찾으러 끝까지감
                parent = current
                if key.num < current.data.num: #좌우어디로 갈지 결정해야하니 키랑 비교
                    current = current.left
                else:
                    current = current.right
        return find, parent, current
    
    #데이터 삭제함수(난이도上)
    def delete(self, key):
        #삭제하려고 할경우 삭제할 노드검색
        if self.root == None:
            return
        found, parent, current = self.find(key)
        if found == False: #삭제대상이없다
            return
        
        #1.삭제대상이 자식이없을때 나자신을 삭제
        if current.left==None and current.right==None: #내가 부모좌우중 어디있었나모름
            if parent.left== current: #내가 부모노드의 좌측에 있었다면
                parent.left=None
            else:                  #내가 부모노드의 우측에 있었다면
                parent.right=None
            return

        #2.자식이 둘中 하나만 있을때
        if current.left != None or current.right != None: #좌우中어디든
            if current.left != None: #좌측에 자식이 있으면 그자식을 가져온다
                parent.left = current.left
            else: #우측에 자식이있다.
                parent.data = current.right
            return
        
        #자식이 둘다있을때 트리전체를 재편
        #삭제될 대상의 우측 서브트리에서 가장 작은대상을 찾아 교체

        #탐색을다시해야한다
        subParent = current #삭제대상
        subCurrent = current.right
        #탐색알 서브트리의 좌측으로. 왜냐면 작은건 좌, 큰건 우로
        


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
