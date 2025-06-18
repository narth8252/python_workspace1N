#0512_3시pm 새문법배우면 재밌을것같아서 배우는거지 용어만 알고있으면됨.

#싱글톤패턴 : 객체를 하나만 만들수 있는 클래스 (대표적ex.DB)
    #우리가 구축할일 거의 없지만 용어만 알아라.(면접시 물어보기좋음)
    #DB커넥션풀을 만들어사용할때: DB연결해서-읽고쓰고-연결끊고 나갈때
        #연결하고 끊는 시간이 읽고쓰는 시간보다 오래걸려서 배보다 배꼽
        #굳이 올때마다 새로만들어서 쓰고 폐기하는것보다
        #연결자를 미리 만들어놓고 돌려쓰자. 
        #식당을 만들어놓고 고객받는것임. 매번 식당만드는게 아니라.
        # → 풀기법:객체 1개만 만들어서 공유 (ex.매니저 class, 스프링,)

#  Singleton 패턴이란?
# 하나  의  톤 → 하나의 인스턴스
#Single + ton = Singleton
# 프로그램 전체에서 하나의 인스턴스만 만들도록 보장하는 디자인 패턴
# → 즉, 클래스의 객체가 하나만 생성되도록 제한함

# 왜 쓰나요?
# 공유 자원이 하나여야 할 때 (예: 설정, DB 연결, 로그 시스템)
# 여러 개 생성되면 비효율적이거나 버그가 날 때

class Singleton:
    __instance = None #객체만들라고하면 None이아닐때만 객체만들고, None이면 객체반환

    @classmethod
    def getInstance(cls):
        if cls.__instance == None: #is None
            cls.__instance = cls.__new__(cls)
                           #클래스를 이용해 인스턴스 만들기라는 문법
            # print("새 인스턴스 생성")
           cls.__init__(cls.__instance)
        return cls.__instance
    
    def display(self): #self필수:인스턴스가 만들어지고, 여기접근하기위한 걸 getInstance를 따로만들어.
        print("************")

    def __init__(self):
        #이미 객체가 존재하면 강제에러 발생
        if Singleton.__instance is not None:
            raise Exception("이클래스는 getInstance()로만 객체를 생성해야 합니다.")
            #Ex부터 가능합니다까지 강제출력

s1 = Singleton.getInstance()
s1.display()

#0512 4시pm
#클래스외부에서 객체 만드는 것들 파이썬이 막을수없음
#다른언어들은 생성자한테도 접근권한있어서 이걸 private으로 만들면 외부에서 객체생성불가
#파이썬 생성자에 이미 __(private)이 붙어있어서 별도로 접근권한을 건드릴수없다.
#편법을 써야. 데이터말고 일을하는 클래스들을 만들때 좋다.
#쓸데없이 객체가 생성/소멸하는걸 방지(과거에 메모리가 작아서 줄이느라 그랬음)
s2 = Singleton()
s2.display()
# b = Singleton()
# print(a is b)  # True → 둘은 같은 객체