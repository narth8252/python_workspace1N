#정수.txt파일읽어와서 평균값 구하기 0509 11:40am
#읽어올 파일의 마지막문장에 커서를 두는것이 출력용이
f = open("정수.txt", "r", encoding='utf-8')  #, encoding='utf-8' 한글읽어올때 오류안나게 인코딩.
lines = f.readlines() 
sum = 0
print(lines)
for line in lines:
    if "\n" in line: #\n이 제거됨
        # line = line.strip()
        line = line[:len(line)-1]

    print(line)
    sum += int(line)
f.close()

print("평균", sum/len(lines))

# s = """
# #문제:이 파일읽어서 평균값출력
# 10
# 20
# 40
# 50
# 4
# 5
# 11
# 12
# 14
# 27 
# #커서위치를 여기다가 두고 py파일에서 출력하면 이쁘게 나옴. if문없어도.(이글있어서 읽어올때 괴상한글출력됨)

# """
# with open('정수.txt', 'w', encoding='utf-8-sig') as f:
#     for a in s:
#         f.write(a)