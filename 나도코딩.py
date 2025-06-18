print("풍선", "나비")
# 불리안 Flase/True
print(not False)  # not 반대임

print("우리집 강아지의 이름인 연탄이이고, 연탄이는 4살이고 연탄이는 어른이예요 True")

animal = "강아지"
name = "연탄"
age = 4
is_adult = age >= 3


print(
    "우리집" + animal + "의 이름은" + name + "고," + name + "는",
    age,
    "살이고" + name + "는 어른이예요",
    is_adult,
)

age = 7
print(
    "우리집"
    + animal
    + "의 이름은"
    + name
    + "고,"
    + name
    + "는"
    + str(age)
    + "살이고"
    + name
    + "는 어른이예요"
    + str(is_adult)
)

# 변수명station, 변수값"사당" "신도림" " 인천공항"순서대로 입력해서 xx행 열차가 들어오고 있습니다. 출력
station = "사당"
print(station + "행 열차가 들어오고 있습니다.")
station = "신도림"
print(station, "행 열차가 들어오고 있습니다.")
station = "인천공항"
print(station, "행 열차가 들어오고 있습니다.")

print(2**3)  # 2^3 = 8
print(5 % 3)  # 나머지구하기 2
print(10 // 3)  # 몫구하기 3
print(4 >= 7)  # False
print(3 == 3)  # True
print(3 + 4 == 7)  # True
print(1 != 3)  # 1은3과같지않다 True
print(not (1 != 3))  # True값의 not반대말 Flase
print((3 > 0) & (3 < 5))  # True
print((3 > 0) or (3 > 5))  # True
print((3 > 0) | (3 > 5))  # True shift+원화=|(or)

number = 2 + 3 * 4  # 14
number = number + 2  # 16
print(number)
number += 2  # 18
print(number)
number *= 2  # 36
print(number)
number /= 2  # 18
print(number)
number -= 2  # 16
print(number)
number = -2  # -2
print(number)
number += 12  # 10
print(number)
number %= 3  # 1
print(number)

print(abs(-5))  # 절대값 5
print(pow(4, 2))  # 4^2 = 4*4 = 16
print(max(5, 12))  # 12
print(min(5, 12))  # 5
print(round(3.14))  # 반올림 3
print(round(4.75))  # 5
print("------------------------")
from math import *  # math 함수안의 *모든것을 쓰겠다.

print(floor(4.99))  # 내림 4
print(ceil(3.14))  # 올림 4
print(sqrt(16))  # 제곱근=루트16=4
print("========================")
from random import *

print(random())  # 0.0~1.0미만의 랜덤값생성
print(random() * 10)  # 0.0~10.0미만의 랜덤값생성 (출력때마다 다른값)
print(int(random() * 10))  # int(소수점떼고 정수만 보여줘라.)0~10*미만*의 랜덤값생성
print(int(random() * 10))  # 소수점떼고 정수만 보여줘라.
print(int(random() * 10))  # 0~10미만의 랜덤값생성
print("-----------------------")
print(int(random() * 10) + 1)  # 1~10이하의 랜덤값 생성
print(int(random() * 10) + 1)
print(int(random() * 10) + 1)
print("======int(random)=============")
print(int(random() * 45) + 1)  # 1~45이하의 랜덤값 생성
print(int(random() * 45) + 1)  # 1~45이하의 랜덤값 생성
print(int(random() * 45) + 1)  # 1~45이하의 랜덤값 생성
print("ˇˇˇrandrange(1, 46)ˇˇˇˇˇˇˇˇ")
print(randrange(1, 46))  # 1~46 미만의 랜덤값 생성
print(randrange(1, 46))  # 1~46 미만의 랜덤값 생성
print("ˇˇˇrandint(1, 45)ˇˇˇˇˇˇˇˇ")
print(randint(1, 45))  # 1~45 를 포함하는 랜덤값 생성

# 월4회 스터디=3번온라인+1번오프라인 모임날짜정하는 프로그램짜기 (랜덤,월별날짜는28일이내,매월1~3일은제외)
# 출력문: 오프라인스터디모임날짜는 매월x일로 선정됨
from random import *

date = randint(4, 28)
print("오프라인 스터디 모임날짜는 매월" + str(date) + "일로 선정되었습니다.")

print("ˇˇˇ문자열ˇˇˇˇˇˇˇˇ")
sentence = "나는 소년입니다."
print(sentence)
sentence2 = "파이썬은 쉬워요"
print(sentence2)
sentence3 = """
나는 초보자고,
파이썬은 어려워요.
"""  # 줄바꿈
print(sentence3)
sentence4 = """나는 초보자고, 파이썬은 랄라블라."""
print(sentence4)

print("ˇˇˇ슬라이싱ˇˇˇˇˇˇˇˇ")
jumin = "990120-1234567"
# 내가 필요한 정보만 가져오는것[0번째부터:몇번째 앞 위치의 값가져올지]
print("성별: ", jumin[7])
print("년: ", jumin[0:2])  # 0번째부터 2번째직전값까지 (0,1번값)
print("월: " + jumin[2:4])  # 0번째부터 2번째직전값까지 (2,3번값)
print("일: ", jumin[4:6])  # 0번째부터 2번째직전값까지 (4,5번값)
print("생년월일: " + jumin[0:6])  # 0번째부터 6번째직전값까지
print("생년월일: " + jumin[:6])  # 0번째부터 6번째직전값까지
print("뒤7자리: " + jumin[7:13])  # 7번째부터 13번째직전값까지
print("뒤7자리: " + jumin[7:])  # 7번째부터 끝까지

print("뒤7자리(뒤에서부터): " + jumin[-7:])  # 맨뒤에서 7번째부터 끝까지

print("ˇˇˇ문자열함수ˇˇˇˇˇˇˇˇ")
python = "Python is Amazing"
print(python.lower())  # 소문자
print(python.upper())  # 대문자
print(python[0].isupper())  # 문자열의 0번째문자가 대문자인가? True
print(len(python))  # 문자열의 길이 17
print(python.replace("python", "java"))  # 문자열에서 "python"문자 찾아서, "java"로 대체

index = python.index("n")  # 문자열에서 n글자가 몇번째에 위치하는지? 5번째(0,1,2,3,4,5)
print(index)
index = python.index("n", index + 1)
# 문자열에서 n글자가 몇번째에 위치하는지인데,
# 직전에찾은 5번째위치에서 +1을한 6째부터 찾는다.(2번째n인 15번째)
print(index)

print(python.find("n"))
print(python.find("java"))
# 문자열에 "java"글자가 없으므로 -1값반환하고 그 뒤로도 출력됨
#  index로 찾으면 error나면서 출력멈춤.
print(python.count("n"))  # 문자열안에 n이 몇번나왔는지count 2번

print("ˇˇˇ문자열 포맷ˇˇˇˇˇˇˇˇ")
# 방법1 %s사용하고 %"원하는말"
print("나는 20살입니다.")
print("나는 %d살입니다." % 20)  # %d는 뒤에있는 %정수값을 d위치에 넣겠다.
print("나는 %s를 좋아해." % "고양이")  # str이니 글자값
print("Apple 은 %c로 시작해요" % "A")  # caracter라서 한글자만 받겠다

print("나는 %s색과 %s색을 좋아해." % ("파란", "빨간"))
# %s 로 바꿔도 출력잘됨.

# 방법2 {}사용하고.fomat(원하는말)
print("나는 {}살입니다.".format(20))
print("나는 {}색과 {}색을 좋아해.".format("파란", "빨간"))
print(
    "나는 {2}색과 {0}색을 좋아해.".format("파란", "빨간", "노란")
)  # format.뒤에 내가쓴것중에 0번째와 2번째걸 넣어출력

# 방법3
print("나는 {age}살이며, {col}색을 좋아해요.".format(age=20, col="빨간"))

# 방법4. 변수선언하고 f"스트링{}
age = 20
col = "빨간"
print(f"나는 {age}살이며, {col}색을 좋아해요.")

print("ˇˇˇˇˇ탈출 문자ˇˇˇˇˇˇˇˇ")
# \n 엔터줄바꿈
print("백문이 불여일견\n백견이 불여일타")
##\" \'는 문장내 따옴표 표기
print('저는 "나도코딩"입니다')  # 할수있지만 보통 ""로 문자열을 감싸므로
print(
    '저는 "나도코딩"입니다'
)  # 저는 "나도코딩"입니다. 라고 출력하고싶을때 ""사이를 문자열로 인식하므로
print("c:\\user\\")
# \r : 뒤에 쓴글자를 맨앞으로 대체 replace
print("Red Apple\rPine")  # PineApple
# \b :백스페이스
print("Redd\bApple")  # RedApple
# \t : 탭처리
print("Red\tApple")  # RedApple

# 문제.비번만들는 프로그램, http://naver.com 에서 생성된 비번 nav51!
# 규칙1.http://는제외 => naver.com
# 규칙2.처음.이후부분 제외 => naver
# 규칙3. 남은글자중 첫3자리(nav)+글자개수(5)+글자내'e'개수+"!"로 구성
url = "http://naver.com"
my_str = url.replace("http://", "")  # 규칙1
# print(my_str)
my_str = my_str[: my_str.index(".")]  # 규칙2.my_str[0:5] 0부터.나오기전까지
# print(my_str)
password = my_str[:3] + str(len(my_str)) + str(my_str.count("e")) + "!"
print("{0}의 비빌번호는{1}입니다.".format(url, password))


print("ˇˇˇˇˇ5-1.리스트[]ˇˇˇˇˇˇˇ")
# 지하철 칸별로 10명, 20명, 30명
# sub1 = 10
# sub2 = 20
# sub3 = 30
sub = [10, 20, 30]
print(sub)
sub = ["유재석", "조세호", "박명수"]
print(sub)
# 조세호가 몇번째 칸에 타고있는가?
print(sub.index("조세호"))  # 0,1,2번째 중에 1

# 담역에서 하하가 담칸에 탐
sub.append("하하")  # 맨뒤에 추가
print(sub)

# 정형돈이 유재석/조세호 사이에 탐
sub.insert(1, "정형돈")  # 0,1번째중에 1번째에 정형돈을 넣음
print(sub)

# 사람을 뒤에서 한명씩 꺼냄
print(sub.pop())  # 맨뒤에 하하가 빠짐
print(sub)

# 같은이름의 사람이 몇명있는지확인
sub.append("유재석")
print(sub)
print(sub.count("유재석"))  # 2

# 정렬
num_list = [5, 4, 1, 2, 3]
num_list.sort()  # [1, 2, 3, 4, 5]
print(num_list)

# 뒤집기
num_list.reverse()  # [5, 4, 3, 2, 1]
print(num_list)

# 모두지우기
num_list.clear()  # []
print(num_list)

# 다양한자료형 믹스
num_list = [5, 4, 1, 2, 3]
mix_list = ["조세호", 20, True]
print(mix_list)  # ['조세호', 20, True]

# 리스트확장
num_list.extend(mix_list)  # [5, 4, 1, 2, 3, '조세호', 20, True]
print(num_list)


print("ˇˇˇˇˇ5-2.딕셔너리{} [.keys()+.values()=.items()]ˇˇˇˇˇˇˇ")
cabinet = {3: "유재석", 100: "김태호"}  # 키가3이고 value가 유재석.
print(cabinet[3])
print(cabinet[100])  # 값이 없는경우에 에러
print(cabinet.get(3))  # 값이 없는경우에 None출력
print(cabinet.get(5, "사용가능"))  # 값이 없지만 사용가능 출력

print(3 in cabinet)  # True
print(5 in cabinet)  # False

# 정수아닌 str도 가능
cabinet = {"A-3": "유재석", "B-100": "김태호"}  # 키가A-3이고 value가 유재석.
print(cabinet["A-3"])
print(cabinet.get("A-5", "사용가능"))  # 값이 없지만 사용가능 출력

# 새손님
print(cabinet)  # {'A-3': '유재석', 'B-100': '김태호'}
cabinet["A-3"] = "김종국"
cabinet["C-20"] = "조세호"
print(cabinet)  # {'A-3': '김종국', 'B-100': '김태호', 'C-20': '조세호'}

# 간손님
del cabinet["A-3"]  # {'B-100': '김태호', 'C-20': '조세호'}
print(cabinet)

# key들만 출력
print(cabinet.keys())  # dict_keys(['B-100', 'C-20'])
# value들만 출력
print(cabinet.values())  # dict_values(['김태호', '조세호'])
# keys, values 함께 출력
print(cabinet.items())  # dict_items([('B-100', '김태호'), ('C-20', '조세호')])

# 목욕탕 닫음
cabinet.clear()
print(cabinet)

print("ˇˇˇˇˇ5-3.튜플:변경x속도빠름ˇˇˇˇˇˇˇ")
menu = ("돈까스", "치즈까스")
print(menu[0])
print(menu[1])
# menu.add("생선까스") 불가

name = "유재석"
age = 50
hobby = "수다"
print(name, age, hobby)
# 튜플로
(name, age, hobby) = ("김종국", 40, "헬스")
print(name, age, hobby)


print("ˇˇˇˇˇ5-4.세트ˇˇˇˇˇˇˇ")
