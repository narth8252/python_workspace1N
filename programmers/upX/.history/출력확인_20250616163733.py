
def solution(n):
    answer = 0
    i = 2 #2부터시작, 첫번째짝수
    for i in range(2, n+1, 2): #2부터시작, 끝값n까지n+1, 2씩증가값(짝수만반복)
        answer += i
    return answer
print(solution(10))