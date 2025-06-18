# """
# https://wikidocs.net/224877
#  좌충우돌, 파이썬으로 자료구조 02-01.정수배열에서 가장큰 두수를 찾기
#      max2 = n
#     return [max1, max2]

# 예시1.
# arr = [3, -1, 5, 0, 7, 4, 9, 1]
# max1 = 3
# max2 = -1

# max1>5    max1=5   max2=3
# max1>0, max2>0  
# max1>7   7보다 안크지. 그러면-> 값을옮기고 젤큰값집어넣어 max1=7 옮기고 max2=5 넣어
# max1>4  
# max1>9  max1=9   max2=7

# 예시2.
# arr = [3, -1, 5, 7, 6, 4, 9, 1]
# 1. max1 = 3    max2 = -1
# 2. 3번째부터 반복
#     만약 max1<arr[2](5)	 max2=max1(3)   max1=arr[2]   max1=5
#     만약 max1<arr[3](7)	 max2=max1(5)   max1=arr[3]   max1=7
#     만약 max1<arr[4](6)
# 	    만약 max2<arr[4]   max2=arr[4](6)   max1=arr[2]   max1=7
#         """

# 문제
# 정수로 이루어진 배열이 주어질 때, 가장 큰 두 수를 찾아 [가장 큰 값, 둘째로 큰 값]을 반환하는 함수를 완성하라.
# 입력: [3, -1, 5, 0, 7, 4, 9, 1], 출력: [9, 7]
# 입력: [7], 출력: [7]
# 여기서는 배열을 순회하면서 직접 비교해 값을 찾는 방식으로 구현해라.

# 배열의 첫 번째와 두 번째 원소를 각각 max1, max2에 대입한다.
# 만약 max2가 max1보다 크다면 두 값을 교환한다.
# 세 번째 원소부터 마지막 원소까지 차례대로 max1, max2와 비교한다.
# 비교하는 원소가 max1보다 크면 max1에 그 원소를 대입하고, max1 값을 max2에 대입한다.
# 그렇지 않고, 그 원소가 max2보다 크면 max2에 대입한다.

def find_max_two(arr: list[int]) -> list[int]:
    """정수 리스트에서 가장 큰 값 두 개를 찾아서 리스트로 반환한다.
    Arguments:
        arr (list): 정수 리스트
    Return:
        list: [가장 큰 값, 둘째로 큰 값]
    """
    if len(arr) < 2:
        return arr
    max1, max2 = arr[:2]
    if max2 > max1:
        max1, max2 = max2, max1
    for n in arr[2:]:
        if n > max1:
            max1, max2 = n, max1
        elif n > max2:
            max2 = n
    return [max1, max2]


arr = [[3, -1, 5, 0, 7, 4, 9, 1], [7]]
for a in arr:
    print(f"{a}에서 가장 큰 두 값: {find_max_two(a)}")



# 0602 am 11:20쌤풀이
def getMax1Max2(arr:list) -> tuple:
  
    """함수설명쓰기
    """
    max1 = arr[0]
    max2 = arr[1]
      #max1이 max2보다 작은경우
    if max1 < max2:
        max1, max2, = max2, max1
    print("max1", max1) #디버깅용 출력, 확인후 제출시는 삭제
    print("max2", max2) #디버깅용 출력, 확인후 제출시는 삭제

    for i in range(2, len(arr)):
        if max1 < arr[i]:
            max2 = max1
            max1 = arr[i]
        elif max2 < arr[i]:
            max2 = arr[i]
        print("max1", max1) #디버깅용 출력, 확인후 제출시는 삭제
        print("max2", max2) #디버깅용 출력, 확인후 제출시는 삭제

# Test code
arr = [3, 4,2,9,8,7,6,11,12,5]
getMax1Max2(arr)