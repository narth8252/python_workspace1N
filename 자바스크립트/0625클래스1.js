class person{
    //생성자
    constructor(name, age) {
        this.name = name; // this는 현재 객체를 가리킴
        this.age = age;
    }

    display() {
        console.log(this.name, this.age); // this는 현재 객체를 가리킴
        // this는 현재 객체를 가리키며, display() 메서드 내에서 this.name은 person.name과 같다.
        // this.생략불가, 람다로 대체불가
        // console.log(`이름: ${this.name}, 나이: ${this.age}`);
        // this는 객체 자신을 가리킨다.
        // 따라서 display() 메서드 내에서 this.name은 person.name과 같다.
        // this.생략불가, 람다로 대체불가
        // (람다=화살표 함수는 this를 바인딩하지 않음,못씀,사실상 함수표현식만 가능)
    }
};
//클래스에 상속도 있고 다 하지만 잘 안씀. 쌤PPT.106p