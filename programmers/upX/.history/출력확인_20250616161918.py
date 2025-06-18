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

#피자나눠먹기1:n명이 주어질때 모든사람이 1조각이상먹기위해 필요한 피자의수
import math
def solution(n):
    pizza = 1
    answer = pizza/7 *n
    return math.ceil(answer)

def solution(n):
    return (n+6)//7

def solution(n):
    return (n-1)//7+1

def solution(n):
    i=0
    while i * 7 < n:
        i += 1
    return i


#피자나눠먹기3
#피자조각수2~10slice, 사람n명이 1slice이상 피자먹으려면 최소몇판시킬지?
def solution(slice, n):
    return((n-1)//slice)+1

#math모듈에서 올림함수ceil()을 사용하자
from math import ceil
def solution(slice, n):
    return ceil(n/slice)

#math모듈에서 내림함수floor()을 사용하자
# from math import floor

#피자나눠먹기2
#6조각피자, n명이 주문한 피자 안남기고 모두 같은조각 먹어야한다면 몇판시켜?
def solution(n):
    pizza = 1
    while True:
        if (p*6) % n > 0:
            p += 1
        if (p*6) % n 