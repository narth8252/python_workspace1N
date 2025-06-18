# f = open("mpg.csv", "r")
# line = f.readlines()
# print(line[:3])

# f = open("mpg.csv", "r")
# line = f.readlines()
# print(line[:3])

#파이썬이 자동닫아주네? with구문사용해보려했더니..
#파이썬 ver.낮을경우에 거듭해서 파일open은 안된다.

with open("mpg.csv", "r") as f:
    lines = f.readlines()
    print(lines[:3])
