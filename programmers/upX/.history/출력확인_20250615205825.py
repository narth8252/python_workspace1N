str1 = "apple banana grape"
# words = str1.split()   #스페이스(기본값) #['apple', 'banana', 'grape']
# str2 = "apple,banana,grape"
# words = str2.split(",") #쉼표 #['apple', 'banana', 'grape']
print(str1.strip().split())  # ['apple', 'banana', 'grape']
# print(words)  # ['apple', 'banana', 'grape']


# arr = [1, 1, 3, 3, 0, 1, 1]
# print(solution(arr))
    

words = ["like", "I" , "python"]
words[0], words[2] = words[2], words[0] #출력순서변경(리스트 자체를 바꿔도됨)
sentence = "* ".join(words) #"공백"힌칸으로 이어줌
print(sentence)  #python* I* like


def solution()