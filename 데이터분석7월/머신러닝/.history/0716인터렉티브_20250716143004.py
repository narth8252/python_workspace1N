#250716 PM2:20
import plotly.graph_objects as go
import pandas as pd
import numpy as np

#차트그리는 함수 하나씩 만들어서 호출
def create_scatter(): #산포도
    print("1.산포도")
    df = pd.DataFrame(
        'x_data':[1,2,3,4,5,6,7,8,9,10],
        'y_data':[1,2,3,4,5,6,7,8,9,10],
        'size_data':[10,10,40,20,50,60,10])

chart1 = go.Scatter(
    x = df["x_data"],
    y = df["y_data"],
    mode='markers',
    marker=dict(size=df("size_data"), color="red", opacity=0.7,
                line = dict(width=1, color='DarkSlateGrey')),
                name="산포도")

#x축에 대한정보 별도생성
layout = go.Layout(title='산점도', xaxis={"title":"X축"},
                   yaxis=dict(title="Y축"),
                   ho)