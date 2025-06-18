#mpg.csv파일 읽어와서 cylinders 개수 8 6 4 종류별 카운트하기
#vs코드 실행폴더에 읽어올파일이 있어야함.
#중복제거함수 drop_duplicates() 
#첫줄빼고 398행
#dict타입추천:
#0509 4시pm 풀이

f = open("mpg.csv", "r")
lines = f.readlines() 
f.close()

lines = lines[1:] #1번방부터 끝까지 복사
print(lines[:4]) #4행(가로줄)까지만 출력
cylinders_count = {} #dict

for line in lines:
    if "\n" in line: #\n이 제거됨
        line = line[:len(line)-1]
        values = line.split(",")
        if values[1] in cylinders_count.keys():
            cylinders_count[values[1]] += 1
        else:
            cylinders_count[values[1]] = 1

print(cylinders_count)
