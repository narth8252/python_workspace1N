
def knapsack(items, w):
    items.sort(key=lambda x:x[1], reverse=True )
    value = 0

    #방법1. 노가다 if문 3번반복하며 누적
    if items[0][1] < w:
        value = items[0][0]
        w -= items[0][1] #잔량구함
    else: #kg당가치=가치/무게
        kg = items[0][0]/items[0][1]
        value += kg * w
        w -= w 
    #    i += 1

    if items[1][1] < w:
        value += items[1][0]
        w -= items[1][1]
    else: #kg당가치=가치/무게
        kg = items[1][0]/items[1][1]
        value += kg * w
        w -= w 

    if items[2][1] < w:
            value += items[2][0]
            w -= items[2][1]
    else: #kg당가치=가치/무게
        kg = items[2][0]/items[2][1]
        value += kg * w
        w -= w #w=0 무게초과이므로 종료

    #방법2. 3번반복하는걸 for문으로 변경
    for i in range(0, len(items)):
        if items[i][1] < w:
            value += items[i][0]
            w -= items[i][1]
        else: #kg당 가치(가치/무게)
            kg = items[i][0]/items[i][1]
            value += kg * w
            w -= w
            break #무게 초과

    return value

w = 50 #이 배낭은 50kg까지 담을수있다
items = [(120,30), (60,10), (100,20)]
print(knapsack(items,w)) #220.0

arr = [1, 1, 3, 3, 0, 1, 1]
print(solution(arr))
    

        