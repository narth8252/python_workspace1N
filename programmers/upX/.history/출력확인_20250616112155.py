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


def solution(num1, num2):
    answer = num1+num2
    return answer

def solution(numbers):
    return[num*2 for num in numbers]

def solution(n, k):
    mok = n // 10
    answer = n*12000+(k-mok)*2000
    return answer

def solution(n, k):
    return 12000 *n +2000 * (k-n//10)

def solution(n,k):
    service = n //10

#n이하 짝수더하기
def solution(n):
    answer = 0
    i = 2
    while i <= n:
        answer+=i
        i+=2
    return answer

#배열의 평균값
def solution(numbers):
    answer = sum(numbers)/len(numbers)
    return answer

import numpy as np
def solution(numbers):
    return np.mean(numbers)

def solttion(numbers):
    return sum(numbers)/len(numbers)

#피자나눠먹기1:n명이 주어질때 