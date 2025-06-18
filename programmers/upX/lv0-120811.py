# 코딩테스트 연습>코딩테스트 입문>중앙값 구하기

"""
중앙값은 어떤 주어진 값들을 크기의 순서대로 정렬했을 때 가장 중앙에 위치하는 값을 의미합니다. 
예를 들어 1, 2, 7, 10, 11의 중앙값은 7입니다. 
정수 배열 array가 매개변수로 주어질 때, 중앙값을 return 하도록 solution 함수를 완성해보세요..

 제한사항
array의 길이는 홀수입니다.
0 < array의 길이 < 100
-1,000 < array의 원소 < 1,000

 입출력 예
array           	result
[1, 2, 7, 10, 11]	7
[9, -1, 0]	        0
"""
#머신러닝에서 중요.
#평균의 함정(최대값이나 최소값때문에 영향) 있어서 평균값보다 중앙값으로 많이 넣음.
#평균값과 중앙값이 가장큰의미
#표준편차와 분산확인 (분산크다=격차가 크다.)
#평균값이 낮아도 분산이 작아야 좋음.
#알고리즘보다 머신러닝이 쉬울수 있음. 정해진 절차,방법이 있어서 반복시킬것임
#주제선정해서 구현하는게 더 중요.
#기획(PL, PM)하고 프로그램하는데 인문학적소양없어서 코딩은잘하는데 못알아먹는 사람多

#잘하면 좋지만
def solution(array):
    sorted_array = sorted(array)
    length = len(sorted_array)
    center = length //2
    median = sorted_array[center]
    return median

"""
1.리스트를 작은값부터 정렬
sorted_numbers = sorted(numbers)
2.리스트 길이를 구하고 가운데 인덱스 찾기
length = len(sorted_numbers)
3.가운데 인덱스 찾기(항상 홀수라고 가정多)
center = length // 2
4.중앙값 리턴
median = sorted_numbers[center]

"""

#다른풀이
def solution(array):
    return sorted(array)[len(array) // 2]
#원본을 바꾸고 싶다 → sort()
#원본은 그대로 두고, 정렬된 새 리스트가 필요하다 → sorted()
#중앙값 문제나 데이터 분석할 땐, 원본 보존이 중요하니까 sorted()를 쓰는 게 더 안전한 선택이야.
#원한다면 reverse, key 옵션도 정리해줄 수 있어.

def solution(array):
    array.sort()
    return array[len(array)//2]

#statistics.median() 기본사용법
# 모듈이름. 함수이름(iterable리스트나 튜플등) : 내부에서 자동정렬
import statistics
result = statistics.median([3,1,2])
print(result)

#기본정렬은 오름차순[1,2,3,..., reverse=True 내림차순정렬[3,2,1]]
#key=함수 정렬기준을 함수로 지정
#각요소에 대해 비교기준지정(문자열길이, 절대값 기준 등) 

#예1.문자열길이로 정렬
words = ['apple', 'kiwi', 'banana']
print(sorted(words, key=len)) #['kiwi', 'apple', 'banana']

#예2-1.절댓값 기준 정렬
nums = [-5, -1, 3, 2]
print(sorted(nums, key=abs)) #[-1,2,3,-5]

#예2-2.기본정렬:오름차순
print(sorted(nums)) #[-5, -1, 2, 3]

#예2-3.내림차순정렬
print(sorted(nums, revers=True)) #[3, 2, -1, -5]

#예3-1.문자열을 대소문자 구분없이 정렬 
names = ['alice', 'Bob', 'charlie']
print(sorted(names, key=str.lower)) #['alice', 'Bob', 'charlie']
                    #모두소문자로바꿔

#예3-2.문자열을 대소문자 구분해 정렬(ASCII기준)
words = ['banana', 'Apple', 'cherry']
print(sorted(words)) #['Apple', 'banana', 'cherry']

#예3-3.문자열을 대소문자 구분해 내림차순 정렬
print(sorted(words, reverse=True))

#예4-1.문자열+숫자 혼합 정렬
items = [('apple', 5), ('banana', 2), ()'cherry', 3)]
#4-2.이름기준(대소문자무시)          [0번째요소:apple]
print(sorted(items, key=lambda x: x[0].lower))
#4-3.숫자 수량기준                  [1번째요소:5]
print(sorted(items, key=lambda x: x[1]))

#예5.lambda없이 함수써도됨
#5-1.이름기준
def get_name(item):
    return item[0] #0번째요소:apple 반환
print(sorted(items, key=get_name))
    #get_num()은 각항목의 1번째요소를 기준으로 정렬하겠다=lambda x:x[1]과 같음
#5-2.숫자 수량기준
def get_num(item):
    return[1]
sorted_items = sorted(items, key=get_num)
print(sorted_items) #[('banana', 2), ('apple', 5), ('cherry', 10)]

#간단한정렬로 한번쓰고 버릴때 lambda x:x[]
#정렬기준이 복잡하거나 재사용 def get_key(...):

#lambda 매개변수:리턴값[요소의인덱스] 이름없는 함수를 만드는법
def add(x):
    return x+1
#같은 기능의 lambda :한줄짜리 간단함수
add = lambda x: x+1 

#lambda : 정렬에서 주로씀 sorted, map, filter같은 함수의 key인자로 씀
#lambda1.숫자리스트에서 짝수만 필터링
nums = [1,2,3,4,5,6]
even = list(filter(lambda x: x %2==0, nums))
           #filter(함수 람다짝수만남겨, 리스트)
print(even) #[2, 4, 6]

#lambda2. 문자열 리스트를 길이순 정렬
words = ['apple', 'kiwi', 'banana', 'fig']
sorted_words = sorted(words, key=lambda x: len(x))
                            #문자열 길이 기준 정렬
print(sorted_words) #['fig', 'kiwi', 'apple', 'banana']
            
#lambda3. 튜플 리스트에서 두번째값 기준 정렬
scores = [('Tom', 80), ('Jane', 95), ('John', 70)]
sorted_scores = sorted(sxores, key=lambda x: x[1])
print(sorted_scores) #[('John', 70), ('Tom', 80), ('Jane', 95)]

#lambda4. map + lambda 로 리스트값 제곱하기
nums = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, nums))
              #map(func, list) → 리스트의 각요소에 함수적용
print(squared) #[1, 4, 9, 16]

#lambda5. 단어리스트를 마지막글자 기준으로 정렬
words = ['sun', 'moon', 'star', 'cloud']
sorted_last = sorted(words, key=lambda x: x[-1])
print(sorted_last) #['cloud', 'moon', 'sun', 'star']
