#json과 흡사함
#C:\Users\Admin\Documents\넘파이,판다스\파이썬데이터분석(배포X)\9차시_백현숙

import pandas as pd

data = {
    'name':['홍길동', '임꺽정', '장길산', '홍경래'],
    'kor':[90, 80, 70, 70],
    'eng':[99, 98, 97, 46],
    'mat':[90, 70, 70, 60],
    }

df = pd.DataFrame(data)
print("타입 : ", type(df))
print(df)

print(df.head(3))

print(df.iloc[0,0])
print(df.loc[0,"name"])

#한컬럼을 통으로 보고싶을때
print(df['name'])
print(df['kor'])

