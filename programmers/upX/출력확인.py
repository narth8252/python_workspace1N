#슬라이싱은 파이썬만되니까, 1번 돌리는거 만들고 while루프로 n번돌리기
#1.왼쪽으로 1번회전하기
def Lrotate(arr): 
    #힘드니까 왼쪽으로 돌리기
    # arr= [1,2,3,4,5]
    # arr= [2,3,4,5,1] #맨앞에 1이 맨뒤로 이동
    # temp = arr[0]
    # arr[0] = arr[1]
    # arr[1] = arr[2]
    # arr[2] = arr[3]
    # arr[3] = arr[4]
    # arr[4] = temp
    temp = arr[0]
    for i in range(1, len(arr)):
        arr[i-1] = arr[i]
    arr[-1] = temp

#3.오른쪽으로 1번회전하기
def  Rrotate(arr): 
    # arr= [1,2,3,4,5]
    # temp = arr[4]
    # arr[4] = arr[3]
    # arr[3] = arr[2]
    # arr[2] = arr[1]
    # arr[1] = arr[0]
    # arr[0] = temp
    
    temp = arr[len(arr)-1]
    for i in range(len(arr)-2, -1, -1):
        arr[i+1] = arr[i] # arr[i-1]하면음수나오니까
    # for i in range(len(arr)-1, -1, -1):
        # arr[i] = arr[i-1] # arr[i-1]하면음수나오니까
    arr[0] = temp   

#2.호출하기
def main(arr, n, direction):
    if direction=="L":
        for i in range(0, n):
            Lrotate(arr)

    else:
        for i in range(0, n):
            Rrotate(arr)

#4.테스트프린트
arr=[1,2,3,4,5,6,7,8,9,10]
main(arr, 5, "L") #n바퀴돌면 원상복귀
print(arr)  #[6, 7, 8, 9, 10, 1, 2, 3, 4, 5]
main(arr, 5, "R") #10바퀴돌면 원상복귀
print(arr)  #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]