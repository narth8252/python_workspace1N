#구글링 iris.csv파일다운받아서 vs코드로 열여서 읽기 0509_2시
#각필드별로 평균구하기.
#데이타는 ,로 구분. 실수라서 int말고 float 쓰면되고
#sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target
#꽃받침  길이,             너비,       꽃잎 길이,        너비,           꽃종류(평균의미는없음,범주형데이터)
#실수받아서 2차원배열, dict
#if안들어왔을때 i를 +1씩증가시켜
#line = lines[i]
#계산먼저 하지말고 
# irisList = [] #파일읽기전에 리스트맹글어:1차원데이터가 들어갈

# f = open("iris.csv", "r")
# slsum = 0
# swsum = 0
# plsum = 0
# pwsum = 0
# line = f.readline()
# if "\n" in line: 
#     line = line[:len(line)-1]



# while line !="":
#     print(type(line))
#     print(line)
#     line = f.readline()


# f.close()
# print("평균", sum/len(lines))

#3시pm쌤풀이
irisList = [] #파일읽기전에 리스트맹글어:1차원데이터가 들어갈
f = open("iris.csv", "r", encoding="utf-8")
lines = f.readlines() 

for i in range(1, len(lines)):
    line = lines[i]
    line = line[:len(line)-1]
    print(i, line)
    values = line.split(",")
    data =[float(values[0]), float(values[1]), float(values[2]), 
           float(values[3])]
    irisList.append(data)
f.close()

for iris in irisList:
    print(iris)
#print(irisList)

result = [0,0,0,0]
for j in range(0, 4):
    for i in range(0, len(irisList)):
        result[j] = result[j] + irisList[i][j]

print(result[0]/150, result[1]/150, result[2]/150, result[3]/150)
#첫줄빼고 150행임
count = len(irisList)
for i in range(0,4):
    print(f"{result[i]/count:.2f}", end="\t"),
print()
