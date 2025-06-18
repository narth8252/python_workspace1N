import pickle
import os
from dataclasses import dataclass
from typing import Callable, Optional

path_delimiter = os.path.sep


def create_id(start=0) -> Callable[[], int]:
    id = start

    def inner():
        nonlocal id
        current = id
        id += 1
        return current

    return inner


@dataclass
class Product:
    id: int
    name: str
    company: str
    price: int


class ProductManager:
    id_gen = create_id()
    products: list[Product]

    def __init__(self):
        self.seed()

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.products):
            raise IndexError("Index out of range")
        return self.products[index]

    def seed(self):
        self.products = [
            Product(ProductManager.id_gen(), "코카콜라", "코카콜라", 2000),
            Product(ProductManager.id_gen(), "펩시", "펩시코", 1900),
            Product(ProductManager.id_gen(), "펩시제로라임", "펩시코", 1900),
            Product(ProductManager.id_gen(), "칠성사이다", "롯데칠성", 1800),
            Product(ProductManager.id_gen(), "트레비", "롯데칠성", 2000),
            Product(ProductManager.id_gen(), "삼다수", "광동제약", 900),
            Product(ProductManager.id_gen(), "칸타타", "롯데칠성", 1800),
            Product(ProductManager.id_gen(), "박카스", "동아제약", 1200),
            Product(ProductManager.id_gen(), "포카리스웨트", "동아오츠카", 1500),
        ]

    def create(self):
        name = input("상품 이름을 입력하세요: ")
        company = input("제조사를 입력하세요: ")
        try:
            price = int(input("가격을 입력하세요: "))
        except Exception:
            print("잘못된 값입니다.")
            return

        self.products.append(Product(ProductManager.id_gen(), name, company, price))
        print("상품을 추가하였습니다.")

    def read(self):
        print("상품 목록")
        for i, product in enumerate(self.products):
            print(
                f"  {i + 1}. 제품 이름: {product.name}, 가격: {product.price}원, 제조사: {product.company}"
            )

    def get_item(self) -> Optional[Product]:
        self.read()
        try:
            choice = int(input("상품 번호를 입력하세요: ")) - 1
            return self.products[choice]
        except Exception:
            return None

    def get_item_by_id(self, id: int) -> Optional[Product]:
        for product in self.products:
            if product.id == id:
                return product
        return None

    def update_price(self) -> Optional[Product]:
        print("상품 가격 수정")
        self.read()
        try:
            choice = int(input("상품 번호를 입력하세요: ")) - 1
            price = int(input("수정할 가격을 입력하세요: "))
        except Exception:
            print("잘못된 입력입니다.")
            return

        self.products[choice].price = price
        print("가격을 수정하였습니다.")
        return self.products[choice]


global_products = ProductManager()


@dataclass
class Item:
    id: int
    label: str
    quantity: int
    price: int
    product_id: int


class ItemManager:
    id_gen = create_id()
    items: list[Item]

    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.items):
            raise IndexError("Index out of range")
        return self.items[index]

    def create(self):
        product = global_products.get_item()
        if not product:
            print("상품이 존재하지 않습니다.")
            return

        label = input("아이템 이름을 입력하세요(기본 상품 이름): ")
        if label == "":
            label = product.name

        try:
            quantity = input("초기 물량을 입력하세요(기본 0): ")
            if quantity == "":
                quantity = 0
            else:
                quantity = int(quantity)
        except Exception:
            print("잘못된 값입니다.")
            return

        self.items.append(
            Item(
                ItemManager.id_gen(),
                label,
                quantity,
                product.price * 1.1,
                product.id,
            )
        )
        print("아이템을 추가하였습니다.")

    def read(self):
        print("아이템 목록")
        for i, item in enumerate(self.items):
            product = global_products.get_item_by_id(item.product_id)
            print(
                f"  {i + 1}. 아이템 이름: {item.label}, 가격: {item.price}원, 재고: {item.quantity}"
            )
            print(f"    {product}")

    def read_sellable(self):
        print("아이템 목록")
        for i, item in enumerate(filter(lambda item: item.quantity > 0, self.items)):
            product = global_products.get_item_by_id(item.product_id)
            print(
                f"  {i + 1}. 아이템 이름: {item.label}, 가격: {item.price}원, 재고: {item.quantity}"
            )
            print(f"    {product}")

    def restock(self):
        print("아이템 재고 추가")
        self.read()
        try:
            choice = int(input("아이템 번호: ")) - 1
            quantity = int(input("추가할 물량: "))
        except Exception:
            print("잘못된 입력입니다.")
            return

        self.items[choice].quantity += quantity
        print("재고를 추가하였습니다.")

    def update_item_price(self, product_id: int, price: int) -> None:
        for item in self.items:
            if item.product_id == product_id:
                item.price = price * 1.1

    def remove_item(self):
        print("아이템 삭제")
        self.read()
        try:
            choice = int(input("아이템 번호: ")) - 1
        except Exception:
            print("잘못된 입력입니다.")
            return

        self.items.pop(choice)
        print("아이템을 삭제하였습니다.")

    def sell_item(self, index: int) -> None:
        if index < 0 or index >= len(self.items):
            print("잘못된 아이템 번호입니다.")
            return

        item = self.items[index]
        if item.quantity == 0:
            print("재고가 없습니다.")
            return

        item.quantity -= 1
        print(f"아이템 {item.label}을(를) 판매하였습니다. 남은 재고: {item.quantity}")


@dataclass
class Sale:
    id: int
    price: int
    item_name: str
    item_id: int


class VendingMachine:
    id_gen = create_id()
    item_manager: ItemManager
    money: int
    sales: list[Sale]

    def __init__(self):
        self.item_manager = ItemManager()
        self.money = 0
        self.sales = []

    def add_sale(self, item: Item) -> None:
        self.sales.append(
            Sale(VendingMachine.id_gen(), item.price, item.label, item.id)
        )

    def read_total_sales(self) -> int:
        return sum([sale.price for sale in self.sales])

    def clear_sales(self) -> None:
        self.sales = []

    def update_item_price(self, updated_product: Product) -> None:
        self.item_manager.update_item_price(updated_product.id, updated_product.price)

    def sell_prompt(self) -> str:
        print("\n자판기")
        self.item_manager.read_sellable()
        return input("아이템 번호를 입력하세요 (종료: -1): ")

    def payment_prompt(self) -> str:
        return input("결제할 금액 (취소: -1): ")

    def sell_loop(self) -> None:
        session_sales = 0

        while True:
            try:
                choice = int(self.sell_prompt())
            except Exception:
                print("잘못된 입력값입니다.")
                continue

            if choice == -1:
                break

            price = self.item_manager[choice - 1].price
            print(f"아이템: {self.item_manager[choice - 1].label}, 가격: {price}원")

            try:
                payment = int(self.payment_prompt())
            except Exception:
                print("잘못된 입력값입니다.")
                continue

            if payment == -1:
                print("결제를 취소합니다.")
                continue

            if payment < price:
                print("결제 금액이 부족합니다.")
                print(f"반환된 금액: {payment}원")
                continue

            if payment > self.money:
                print("자판기에 잔액이 부족합니다.")
                print("많은 불편을 드려 죄송합니다.")
                print(f"반환된 금액: {payment}원")
                continue

            change = payment - price
            self.money += price
            self.item_manager.sell_item(choice - 1)
            session_sales += price
            self.add_sale(self.item_manager[choice - 1])
            print("결제가 완료되었습니다.")
            print(f"잔돈: {change}원")

        print(f"오늘의 판매 총액: {session_sales}원")

    def sell(self):
        print("아이템 판매")
        self.read_sellable()
        try:
            choice = int(input("아이템 번호: ")) - 1
        except Exception:
            print("잘못된 입력입니다.")
            return

        self.items[choice].quantity -= 1
        print("아이템을 판매하였습니다.")

    def product_prompt(self) -> str:
        print("\n상품 관리")
        print("1. 상품 보기")
        print("2. 상품 추가")
        print("3. 상품 가격 수정")
        print("4. 메인 메뉴로 돌아가기")
        return input("> ")

    def product_loop(self) -> None:
        while True:
            choice = self.product_prompt()

            if choice == "1":
                global_products.read()
            elif choice == "2":
                global_products.create()
            elif choice == "3":
                updated_product = global_products.update_price()
                if updated_product:
                    self.update_item_price(updated_product)
            elif choice == "4":
                break
            else:
                print("잘못된 입력입니다.")

    def item_prompt(self) -> str:
        print("\n아이템 관리")
        print("1. 아이템 보기")
        print("2. 아이템 추가")
        print("3. 아이템 재고 추가")
        print("4. 아이템 제거")
        print("5. 메인 메뉴로 돌아가기")
        return input("> ")

    def item_loop(self) -> None:
        while True:
            choice = self.item_prompt()
            if choice == "1":
                self.item_manager.read()
            elif choice == "2":
                self.item_manager.create()
            elif choice == "3":
                self.item_manager.restock()
            elif choice == "4":
                self.item_manager.remove_item()
            elif choice == "5":
                break
            else:
                print("잘못된 입력입니다.")

    def vending_machine_prompt(self) -> str:
        print("\n자판기 관리")
        print("1. 자판기 잔액")
        print("2. 자판기 잔액 추가")
        print("3. 인출")
        print("4. 총 판매 금액")
        print("5. 판매 내역")
        print("6. 판매 내역 초기화")
        print("7. 메인 메뉴로 돌아가기")
        return input("> ")

    def vending_machine_loop(self) -> None:
        while True:
            choice = self.vending_machine_prompt()
            if choice == "1":
                print(f"자판기 잔액: {self.money}원")
            elif choice == "2":
                try:
                    amount = int(input("추가할 금액: "))
                except Exception:
                    print("잘못된 입력입니다.")
                    continue

                self.money += amount
                print(f"자판기 잔액이 {amount}원 추가되었습니다.")
            elif choice == "3":
                try:
                    amount = int(input("인출할 금액: "))
                except Exception:
                    print("잘못된 입력입니다.")
                    continue

                if amount > self.money:
                    print("잔액이 부족합니다.")
                    continue

                self.money -= amount
                print(f"{amount}원이 인출되었습니다.")
            elif choice == "4":
                print(f"총 판매 금액: {self.read_total_sales()}원")
            elif choice == "5":
                print("판매 내역")
                for i, sale in enumerate(self.sales):
                    print(f"  {i + 1}. {sale.item_name} {sale.price}원")
            elif choice == "6":
                self.clear_sales()
                print("판매 내역이 초기화되었습니다.")
            elif choice == "7":
                break
            else:
                print("잘못된 입력입니다.")

    def save(self) -> None:
        path = input("저장할 파일 경로를 입력하세요: ")

        with open(path, "wb") as f:
            save_binary = {
                "products": global_products,
                "item_manager": self.item_manager,
                "money": self.money,
                "sales": self.sales,
            }
            pickle.dump(save_binary, f)
        print("정보를 저장하였습니다.")

    def load(self) -> None:
        path = input("불러올 파일 경로를 입력하세요: ")
        if not os.path.exists(path):
            print("파일이 존재하지 않습니다.")
            return

        with open(path, "rb") as f:
            info = pickle.load(f)
            global_products.products = info["products"].products
            self.item_manager.items = info["item_manager"].items
            self.money = info["money"]
            self.sales = info["sales"]

            global_products.id_gen = create_id(len(global_products.products))
            ItemManager.id_gen = create_id(len(self.item_manager.items))
            VendingMachine.id_gen = create_id(len(self.sales))
            ProductManager.id_gen = create_id(len(global_products.products))

        print("정보를 불러왔습니다.")

    def prompt(self) -> str:
        print("\n자판기 관리 프로그램")
        print("1. 판매")
        print("2. 상품 관리")
        print("3. 아이템 관리")
        print("4. 자판기 관리")
        print("5. 정보 저장")
        print("6. 정보 불러오기")
        print("7. 종료")
        return input("> ")

    def main_loop(self) -> None:
        while True:
            choice = self.prompt()

            if choice == "1":
                self.sell_loop()
            elif choice == "2":
                self.product_loop()
            elif choice == "3":
                self.item_loop()
            elif choice == "4":
                self.vending_machine_loop()
            elif choice == "5":
                self.save()
            elif choice == "6":
                self.load()
            elif choice == "7":
                break
            else:
                print("잘못된 입력입니다.")


def main():
    vending_machine = VendingMachine()
    vending_machine.main_loop()


if __name__ == "__main__":
    main()
