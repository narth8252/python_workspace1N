"""
# 코딩테스트 연습 > 카드 뭉치 159994
코니는 영어 단어가 적힌 카드 뭉치 두 개를 선물로 받았습니다. 
코니는 다음과 같은 규칙으로 카드에 적힌 단어들을 사용해 
원하는 순서의 단어 배열을 만들 수 있는지 알고 싶습니다.

원하는 카드 뭉치에서 카드를 순서대로 한 장씩 사용합니다.
한 번 사용한 카드는 다시 사용할 수 없습니다.
카드를 사용하지 않고 다음 카드로 넘어갈 수 없습니다.
기존에 주어진 카드 뭉치의 단어 순서는 바꿀 수 없습니다.
예를 들어 첫 번째 카드 뭉치에 순서대로 ["i", "drink", "water"], 
두 번째 카드 뭉치에 순서대로 ["want", "to"]가 적혀있을 때 
["i", "want", "to", "drink", "water"] 순서의 단어 배열을 만들려고 한다면 
첫 번째 카드 뭉치에서 "i"를 사용한 후 
두 번째 카드 뭉치에서 "want"와 "to"를 사용하고 
첫 번째 카드뭉치에 "drink"와 "water"를 차례대로 사용하면 
원하는 순서의 단어 배열을 만들 수 있습니다.

문자열로 이루어진 배열 cards1, cards2와 원하는 단어 배열 goal이 매개변수로 주어질 때, 
cards1과 cards2에 적힌 단어들로 goal를 만들 있다면 "Yes"를, 
만들 수 없다면 "No"를 return하는 solution 함수를 완성해주세요.

  입출력 예
cards1	                cards2	            goal	                            result
["i", "drink", "water"]	["want", "to"]	["i", "want", "to", "drink", "water"]	"Yes"
["i", "water", "drink"]	["want", "to"]	["i", "want", "to", "drink", "water"]	
cards1에서 "i"를 사용하고 cards2에서 "want"와 "to"를 사용하여 "i want to"까지는 
만들 수 있지만 "water"가 "drink"보다 먼저 사용되어야 하기 때문에 
해당 문장을 완성시킬 수 없습니다. 따라서 "No"를 반환합니다.
"""
def solution(cards1, cards2, goal):
    answer = ''
    i=0
    j=0
    # k=0
    for k in range(0, len(goal)):
        #and연산은 양쪽다True일때 True, 어느한쪽이 False이면 다른쪽 연산안해도됨
        #shortcut?회로 앞 수식먼저판단후 그결과가 False이면, 뒤의수식은 건너뜀
        #그래서 수식의 순서를 바꾸면 안됨. i값을 증가시켜서 밖으로 못나가게 막아야함
        if i<len(cards1) and cards2[i] == goal[k]:
            i+=1
            k+=1
        if j<len(cards2) and cards2[j] == goal[k]:
            j+=1
            k+=1 #카드1,2는 확인할길이 없으니 골이 증가했나 확인
        else: #둘다아니면 순서어긋남
            return "NO"
    return "Yes"

print(solution(["i", "drink", "water"],
               ["want", "to"],
               ["i", "want", "to", "drink", "water"]))


#merge(병합)정렬 
"""
DB없을때 파일로 작업 - 갱신(쌤 정보처리기사시험때 多)

  인사파일                             수정파일
1  홍길동  2024-03-09   ....       1 D 2025-06-16
                                        101 I 장길산 2025-06-12
"""
"""
i <- a꺼   a[i] == b[j]   c.append(a[i])   i+=1  j+=1
j <- b꺼   a[i] < b[j]    a[i]를 내보낸다. i+=1
              a[j] > b[i]    b[j]를 내보낸다. j+=1

              마지막은 남은배열을 모두 C로 내보내면 된다.
"""
#이미 정렬된 배열을 합쳐 정렬
a = [5, 7, 9, 11, 12, 23, 27,34]
b = [3,4,5,7,9,11,13,19,23,27,27,33,34]
# c = [3,4,5,7,9,11,12,13,19,23,27,33,34]

def merge(a, b):
    c = []
    i = 0
    j = 0
    while i <len(a) and j < len(b):
        if a[i] == b[j]:
            c.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            c.append(a[i])            
            i += 1
        else:
            c.append(b[j])
            j += 1

    while i < len(a):
        c.append(a[i])
        i += 1
    
    while j < len(b):
        c.append(b[i])
        j += 1
    
    return c

print( merge(a,b) ) # c = [3,4,5,7,9,11,12,13,19,23,27,33,34]


