#클래스_야구게임_GameMain.py 0508 5시pm
from GameData import Baseball #from 파일명에서 import 클래스명을 끌어옴

class GameMain:
    def __init__(self):
        self.gameList = []

    def start(self):
        while True: #메뉴적으니까 여기다 쓸께
            print("1.야구게임시작")
            print("2.통계")
            print("0.종료")
            sel = input("선택 :")
            if sel =="1":
                self.gameStart()
            elif sel =="2":
                self.showStatistics()
            else:
                return

    def gameStart(self):
            b = Baseball()
            b.start()
            self.gameList.append(b)

    def showStatistics(self):
        for b in self.gameList:
            print
            for item in b.personList:
                print(item["person"], item["strike"],
                      item["ball"], item["out"], b.count)
    #dict타입이라 출력한거

if __name__=="__main__":
    g = GameMain()
    g.start()