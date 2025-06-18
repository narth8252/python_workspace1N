str1 = "apple banana grape"
# words = str1.split()   #스페이스(기본값) #['apple', 'banana', 'grape']
# str2 = "apple,banana,grape"
# words = str2.split(",") #쉼표 #['apple', 'banana', 'grape']
print(str1.strip().split())  # ['apple', 'banana', 'grape']
# print(words)  # ['apple', 'banana', 'grape']


# arr = [1, 1, 3, 3, 0, 1, 1]
# print(solution(arr))
    

words = ["I", "like", "python"]
sentence = " ".join(words)
print(sentence)  # I like python
