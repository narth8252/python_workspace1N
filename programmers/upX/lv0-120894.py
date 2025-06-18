def solution(numbers):
    #배열"
    num_list=["zero","one","two","three","four",
         "five", "six","seven","eight","nine"]
    temp=""
    result=""
    for n in numbers:
        temp = temp + n
        if temp in num_list:
            i = num_list.index(temp)
            temp=""
            result += str(i)
            
    # answer = 0
    return int(result) #숫자로 달래
               