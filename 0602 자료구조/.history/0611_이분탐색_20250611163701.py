#0611 pm2:25 좌충우돌,파이썬으로자료구조 09이진 탐색 트리
#https://wikidocs.net/195269
# https://wikidocs.net/233716  Concept of Algorithms with Python 5-10. Search : Binary Search
# https://good.oopy.io/algorithms/binary_search  이분탐색 쉽게 푸는 템플릿
# https://code-angie.tistory.com/3 알고리즘/이분탐색/이진탐색 (Binary Search)
# 백준 이분탐색 글 https://www.acmicpc.net/blog/view/109
# 백준 단계별 이분탐색 https://www.acmicpc.net/step/29
# 프로그래머스 이분탐색 https://school.programmers.co.kr/learn/courses/30/parts/12486
#https://kangworld.tistory.com/65  이진탐색(이분탐색, Binary Search)코드와 시간복잡도
# 
"""
1. 이분 탐색의 조건
반드시 오름차순으로 정렬된 상태에서 시작해야 한다.
결정 문제(Decision Problem)의 답이 이분적일 때 사용할 수 있는 탐색 기법입니다. 
이때 결정 문제란 답이 Yes or No인 문제를 의미하며 
(이분 탐색 문제에서는) 보통 1개의 parameter를 가집니다.
많은 최적화 문제는 이분 탐색으로 풀 수 있습니다. 
최적화 문제란 어떤 조건(Check(x))을 만족하는 x의 최댓값 또는 최솟값을 찾는 문제를 말합니다. 
이 경우 Check(x)가 x에 대해 이분적이면 이분 탐색을 사용할 수 있습니다.
2. 이분 탐색 알고리즘
시간복잡도: O(logN)
반복문과 재귀 두 가지 방법을 사용할 수 있다.
자료를 오름차순으로 정렬한다.
자료의 중간값(mid)이 찾고자 하는 값(target)인지 비교한다.
mid 값이 target과 다르다면 대소관계를 비교하여 탐색 범위를 좁히고, target과 mid 값이 같을 때까지 아래 조건에 따라 2번과 3번을 반복한다.
          ⓐ target이 mid 값 보다 작으면 end를 mid 왼쪽 값으로 바꿔준다. (절반의 왼쪽 탐색)

          ⓑ target이 mid 값 보다 크면 start를 mid 오른쪽 값으로 바꿔준다. (절반의 오른쪽 탐색)

a = [1,2,3,4,5,6,7,8,9,10]
key = 8

1. (0+9)//2중간값 = 4번방   5   값이
2. (6+9)//2 = 7번방      8   두번 비교하고 찾기

없을 때
key=13
1. (0+9)//2   -  5
2. (6+9)//2   -  7번방이 8
3. (8+9)//2   -  8번방이 9
4. (9+9)//2   -  9번방이 10
5. (10+9)//2   -  not found
 
left = 0
right = 9

mid = (left + right) // 2
만일 값이 키값보다 크면 a[mid] > key      left = mid+1
만일 값이 키값보다 작으면 a[mid] < key   right = mid-1
찾았던지 left <= right 동안만
"""

def solution(a, key, left, right):
    a.sort()
    if left > right:
        return "Not Found"
    mid = (left + right) // 2
    if a[mid] == key:
        str = f"key값 {key}는 {mid}번 방에 있습니다."
        return str
    elif a[mid] > key:
        right = mid - 1
    elif a[mid] < key:
        left = mid + 1
    
    return solution(a, key, left, right) 

# 강사님 풀이
def binearSearch(arr, key): # 배열, 키값
    #1. 정렬
    arr.sort()
    left = 0
    right = len(arr)-1
    
    while left <= right:
        mid = (left+right) // 2
        if arr[mid] > key:
            right = mid - 1
        elif a[mid] < key:
            left = mid + 1
        elif a[mid] == key:
            return mid
    return -1

a = [1,2,3,4,5,6,7,8,9,10]
key = 8

print(solution(a, key, 0, len(a)-1))
print(solution(a, 13, 0, len(a)-1))
print(binearSearch(a, key))
print(binearSearch(a, 13))

#이분탐색 함수(반복문)
def binary_search(target, data):
  data.sort()
  start = 0         #맨첨위치
  end = len(data)-1 #만끝위치

  while start <= end:
    mid = (start + end) // 2 #중간값

    if data[mid] == target:
      return mid     #target 위치 반환

    elif data[mid] > target:
      end = mid - 1   #target이 작으면 좌측 더 탐색
    
    else:
      start = mid + 1 #target이 크면 우측 더 탐색
  return

#이분탐색 함수(재귀호출)
def binary_search(target, start, end):
  if start > end:         #범위를 넘어도 못찾으면 -1반환
    return -1

  mid = (start + end) // 2 #중간값

  if data[mid] == target:  #중간값이 target과 같으면 mid반환
    return mid

  elif data[mid] > target:  #target이 작으면 좌측 더 탐색
    end = mid - 1
  
  else:                     #target이 크면 우측 더 탐색
    start = mid + 1

  return binary_search(target, start, end) #줄어든 범위를 더 탐색

def solution(target, data):
  data.sort() #오름차순정렬필수
  start = 0
  end = len(data) - 1
  return binary_search(target, start, end)
