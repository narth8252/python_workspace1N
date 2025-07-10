

import pandas as pd 

data = {
    'passenger_code':['A101', 'A102', 'A103', 'A101', 'A104', 'A101', 'A103'],
    'target':['광주', '서울', '부산', '광주', '대구', '광주', '부산'],
    'price':[25000, 27000, 45000, 25000, 35000, 27000, 45000]
}

df = pd.DataFrame(data)
print(df)

print("중복된 데이터")
col = df['passenger_code'].duplicated() #True/False로 보여줌
print(col) #중복된 행렬리스트 줌.
"""
  passenger_code target  price
0           A101     광주  25000
1           A102     서울  27000
2           A103     부산  45000
3           A101     광주  25000
4           A104     대구  35000
5           A101     광주  27000
6           A103     부산  45000
중복된 데이터
0    False
1    False
2    False
3     True
4    False
5     True
6     True
"""

print("중복된 삭제후")
df2 = df.drop_duplicates() #삭제된데이터반환, 전체필드가 완전히 일치하는 데이터만 삭제
print(df2) #중복된 행렬리스트 줌.

print("서브셋 사용하기")
df3 = df.drop_duplicates(subset=['passenger_code'])
print(df2)