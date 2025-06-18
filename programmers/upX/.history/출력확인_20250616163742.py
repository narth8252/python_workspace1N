def solution(n):
    answer = 0
    i = 2 
    for i in range(2, n+1, 2): 
        answer += i
    return answer

print(solution(10))