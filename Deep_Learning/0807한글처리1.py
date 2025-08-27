#250807 딥러닝복습_한글처리 코드복잡
#자바 환경
#cmd 관리자권한 
#conda activate mytensorflow
#pip install konlpy (잘안되면 colab에 이 줄만 치면 끌어다 쓸수있음)
from konlpy.tag import Okt
text = "한글자연어처리는 매우 어렵다.ㅜ.ㅜ 네이버 영화평론 분석을 해보자"

okt = Okt()
print("--- 형태소 분석 ---")
print(okt.morphs(text))

print("--- 품사태깅 ---") #한글을 쪼개서 품사 뽑아냄. 자연어=명사
print(okt.pos(text))

print("--- 명사추출 ---") #차트그릴때 주로 사용
print(okt.nouns(text))

print("--- 정규화 ---") #줄임말이나 인터넷용어 변환하려고 만들어놓은 모듈이 있음
print(okt.normalize(text))

print("--- 어간추출 ---") #동사, 형용사, 원형복원
print(okt.morphs(text, stem=True))
