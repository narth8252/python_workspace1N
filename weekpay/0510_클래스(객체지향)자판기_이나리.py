#0510 안수현님, 오은택님 기프티콘(이나리)
class vendingMachine: 
  def __init__(self):
    self.americano_stock = 3
    self.latte_stock = 2
    self.coke_stock = 3

    self.americano_price = 1500
    self.latte_price = 2000
    self.coke_price = 1000

    self.money = 0

  def menuDisplay(self):
      print("-------메뉴---------")
      print("1.아메리카노",self.americano_price,"원 (재고", self.americano_stock,"잔)") 
      print("2 라떼",self.latte_price,"원 (재고", self.latte_stock,"잔)")
      print("3.콜라",self.coke_price,"원 (재고", self.coke_stock,"잔)")
      print("4.나가기 ")
      print("--------------------")

  def insert_money(self):
      amount = int(input("돈을 넣으세요(숫자만 입력): "))
      self.money += amount
      print(f"잔액 : {self.money}원\n")
      
  def choice_menu(self):
   
    while True:
      self.menuDisplay() #메뉴먼저보여줘
      choice = input("메뉴번호선택:")
      self.insert_money()
      if choice == "1":    

        if self.americano_stock > 0:
            if self.money >= self.americano_price:
                print("아메리카노 나왔습니다")
                self.americano_stock -= 1
                self.money -= self.americano_price
            else:
              print("잔액부족")
        else:
          print("아메리카노 품절")
        
      elif choice == "2":
        if self.latte_stock > 0:
          if self.money >= self.latte_price:
            print("라떼 나왔습니다.")
            self.latte_stock -= 1
            self.money -= self.latte_price
          else:
            print("잔액부족")
        else:
          print("라떼 품절")

      elif choice == "3":          
        if self.coke_stock > 0:
          if self.money >= self.coke_price:
            print("콜라 나왔습니다.")
            self.coke_stock -= 1 
            self.money -= self.coke_price
          else:
            print("잔액부족")
        else:
          print("콜라 품절")

      elif choice == "4":
        print(f"\n잔액 {self.money}원입니다. 안녕히가세요.")
        break

      else:
        return
      
vm = vendingMachine()
vm.choice_menu()
