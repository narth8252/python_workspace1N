# 코딩테스트 연습>코딩테스트 입문>삼각형의 완성조건 (1)
"""
선분 세 개로 삼각형을 만들기 위해서는 다음과 같은 조건을 만족해야 합니다.
가장 긴 변의 길이는 다른 두 변의 길이의 합보다 작아야 합니다.
삼각형의 세 변의 길이가 담긴 배열 sides이 매개변수로 주어집니다.
세 변으로 삼각형을 만들 수 있다면 1, 만들 수 없다면 2를 
return하도록 solution 함수를 완성해주세요.

sides	    result
[1, 2, 3]	    2
[3, 6, 2]	    2
[199, 72, 222]	1
가장 큰변인3이 나머지 두변의 합3과 같으므로 삼각형을 완성할 수없습니다. 
따라서 2를 return합니다.
"""
#풀이 1.오름차순정렬 list.sort() or sorted()
#     2.a+b>c 이면 return 1, 아니면 return 2
def solution(sides):
    sides = sorted(sides) #원본은 두고, 정렬된 새로운 리스트를 반환
    #sides.sort()  # 리스트 자체를 오름차순 정렬해 바로 사용
    a, b, c = sides # 각 변 길이를 a, b, c에 할당
    if a+b > c:
        return 1
    else: 
        return 2
"""
"""
sides = [3, 1, 2]
# sorted 사용 예
new_sides = sorted(sides)
print(new_sides)  # 출력: [1, 2, 3]
print(sides)      # 원본은 그대로: [3, 1, 2]

# sort 사용 예
sides.sort()
print(sides)      # 출력: [1, 2, 3]

#다른풀이
def solution(sides):
    return 1 if max(sides) < (sum(sides) - max(sides)) else 2