#파일을 읽기로 열때는 파일이 존재해야한다.

f = open("데이터파일.txt", "r")
data = f.read() #파일을 통으로 읽는다.
print(data)
f.close()

f = open("데이터파일3.txt", "r")
data = f.read() #파일을 통으로 읽는다. str타입으로
print(type(data))
f.close()  #파일을연다.파일포인터-파일읽을 위치값이 맨뒤에가있다.

f = open("데이터파일3.txt", "r") #str임.
line = f.readlines() #반환값이 list타입이다.
print(type(line))
print(line)
f.close()

#실행시 명령프롬프트창에 ↑ 치면 최근에친 dir 나옴
#출력창에 "dir"실행하고 "type 데이터파일3.txt"실행하면 
#i= 1i= 2i= 3i= 4i= 5i= 6i= 7i= 8i= 9i= 10

f = open("./doit/데이터파일4.txt", "r") #str임.
line = f.readline() #반환값이 list타입이다.
while line !="":
    print(type(line))
    print(line)
    line = f.readline()
f.close()
