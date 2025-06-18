#쌤손코딩 0509 1시pm 
    #score.txt에서 한줄을 읽어온다. ["홍길동,100,90,80"\n]
    #\n 삭제: for line in lines:
                # if "\n" in line:
                    # line = line[:len(line)-1]
    #s = "홍길동,100,90,80"
    #s.strip()으로 공백제거하고
    #.split(",")콤마기준으로 분리해 리스트반환. 
    # "홍길동,100,90,80"은 ['홍길동','100','90','80']로 분할

#2시pm 풀이
#cp949코드에러: 한글못읽어옴→vs코드는utf-8로 읽어옴.
#,encoding='utf-8' 한글읽어올때 오류안나게 인코딩.
#cp949(윈도우방식), utf-8(표준방식,VS코드방식)
f = open("score.txt", "r", encoding='utf-8')  
lines = f.readlines() 
for line in lines:
    if "\n" in line: #\n이 제거됨
        line = line[:len(line)-1]
        
    words = line.split(",")
    print(words)
    name = words[0]
    kor = int(words[1])
    eng = int(words[2])
    mat = int(words[3])
    tot = kor+eng+mat
    avg = tot/3
    print(name, kor, eng, mat, tot, avg)

f.close()

# # print("평균", sum/len(lines))
