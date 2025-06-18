#0611 pm3시 좌충우돌,파이썬으로자료구조 06해시테이블 구현하기
#https://wikidocs.net/193049
"""
해시 테이블은 언어에 따라 해시 맵, 사전 등으로 부른다. 
해시 테이블은 키(key)와 값(value)으로 구성된 자료 구조로, 
데이터를 빠르게 검색하고 저장할 수 있다. 
여기서 중요한 것은 해시 함수다. 
해시 함수는 입력된 키를 통해 고유한 해시 값을 생성하며, 
이 값을 이용해 테이블에 값을 저장하거나 검색한다.
분리연결법이 개방주소법보다 나음

# 해시테이블 개념과 텍스트로 표현한 그림
크기가 8인 해시테이블이 있고, key에 대해 해시함수로 인덱스계산
| 인덱스 (Index) | 저장된 값(Value) |
|----------------|------------------|
| 0              | apple            |
| 1              | None             |
| 2              | orange           |
| 3              | None             |
| 4              | banana           |
| 5              | None             |
| 6              | None             |
| 7              | grape            |

# 텍스트로 표현한 해시테이블
Index | Value
------+---------
  0   | apple
  1   | 
  2   | orange
  3   | 
  4   | banana
  5   | 
  6   | 
  7   | grape

# 해시 충돌이 있는 경우 (예: 체이닝)
인덱스 2에 여러 값이 저장되는 경우(체이닝) 예시
Index | Values (Linked List)
------+----------------------
  0   | apple
  1   | 
  2   | orange -> melon -> kiwi
  3   | 
  4   | banana
  5   | 
  6   | 
  7   | grape

# 해시테이블 구조

- 각 인덱스는 버킷(bucket)에 해당하며,  
- 버킷에는 하나 또는 여러 데이터가 저장됩니다.  
- 위 예시는 링크드 리스트 사용해 충돌 시 여러 값을 저장하는 방법을 나타냅니다.

---

필요하시면 해시 함수나 충돌 처리 방식 등도 설명해 드릴 수 있습니다. 언제든 문의해 주세요.
값 ====> 컴퓨터 메모리로 맵핑
 맵핑함수 - 해쉬함수: 문자열 → ascii 코드로 만들어서 다 더한후
                            충돌나면 안되니 수식통해 특정값 나오게 한다음
                            %키의 전체개수 나머지
값 => 해쉬함수 ==> 해쉬테이블에 저장(특정메모리로 이동)
                bucket배열
                배열만, 배열과 링크드리스트, 링크드리스트로만으로 구성가능

school  ========> 23432%100 32 → bucket[32]
rain    ========> 33454%100 54 → bucket[54] 
rainbow ========> 23412%100 12 → bucket[12]
bucket의 크기는 전체 key값 개수1/2 ~ 1/3
bucket[100]

collison - 충돌, 배열 bucket[54]  →   →   → 
"""
# 1.해쉬함수 만들기
# school 넣으면 - 각 단어별로 문자의unicode 만들고(파이썬 ord('a')함수), 총합구하기
"""
  1.해시테이블 구축하기
  head[0] → ("school"|??) → (|None)
  head[1] → ("rain"|??) → (|None)
  head[2] → ("desk"|??) → ("chair"|?) → (|None)
  
"""
#데이터타입만
class Node:
  def __init__(self, data=None):
    self.data = data
    self.next = None
 
  def __str__(self): #기본연산자 오버라이딩, 객체를 print할때 이연산자를 overring해주면 
                     #원하는대로 출력
                     #안하면 객체 주소값이 출력됨

    return f"{self.data} {self.next}"

# head->(|)-> 
#
# bucketList[0] -> (|)  
# bucketList[1] -> (|) 
# bucketList[2] -> (|) 

class HashTable:
  def __init__(self, cnt=10):
    self.cnt = cnt #10개보다 작으면 최소한 bucket개수를 10개는 만들자
    if self.cnt<10:
      self.cnt = 10
    self.bucketList = [None]*cnt
    #앞에 완충장치용으로 노드를 미리 만들어서 붙여놓자
    self.bucketList = [ Node() for i in range(0, self.cnt)]
                    #for문대신 컴프리헨션 써본다.
    for i in range(0, 5):
      print(self.bucketList[i])
    print(self.bucketList[:5]) #5개까지만 출력되나 확인해보자

  def getHash(self, key): #해쉬함수붙이기
    total = 0
    for k in key:
      total += ord(k)
    return total%self.cnt #100으로 나눠야함
  
  #해쉬테이블 구성함수 만들기 ->기본적으로 해쉬테이블 구축해야 검색가능
  #처음에 해쉬테이블 구축시간이 많이 걸림
  #실제로 mysql안에 옵티마이저 Nested loop join(for문 돌림), Hash join(해쉬테이블)
  #예:Mysql의 join이 외부에서 inner join, outer join
  #hash join은 데이터가 대용량이고 배치처리(일감한번에몰아처리) ↔ 온라인처리(실시간처리)
  #온라인처리(실시간처리): 보통의 경우는 실시간처리(ex.고객이 통장개설)
  #                   → 이때는 nested loop join이 빠르다.
  #trade off : 균형잡기 - 다양한 알고리즘중에 적당한걸 선택해야한다.

# bucketList[0] -> (|)  
# bucketList[1] -> (|) 
# bucketList[2] -> (|) 

  def createTable(self, dataList=None):  #생성자에서 해도되는데 코드길어지니 함수별도제작
                          #생성자에서 호출하면 됨
    #줄때 외부에서 튜플로 줄것임 [("book"),("school"),("rain")]
    for item in dataList:
      #1 해쉬값 가져오기
      key = self.getHash(item) #10개의 bucket중에 어디에 들어갈 것인가?
      # print(item, key)
      #2 키값저장된 Node만들어 노드앞쪽에 붙인다
      bucket = Node(item)
      #3 bucketList의 앞head에 붙인다.
      bucket.next = self.bucketList[key].next   # bucketList[5] ->(|) ('pool') ->('schOol'|None)->
      self.bucketList[key].next = bucket
      # print(self.bucketList[key].next)
  
  def printList(self):
    for item in self.bucketList:
      trace = item.next
      while trace != None:
        print(trace.data, end="=>")
        trace = trace.next
      print()

hash = HashTable()
hash.createTable([("school"), ("desk"), ("chair"), ("rain"), ("survey"),
                  ("school"), ("Home"), ("doll"), ("java"), ("python"),
                   ("java"), ("html"), ("javascript")
                  ])
hash.printList()

# print( getHash("a"))
# print( getHash("korea"))
# print( getHash("school"))
#head = [None]*100 100개 만들고 시작하면 됨
