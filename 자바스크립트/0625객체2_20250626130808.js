let person = {
    name: "홍길동",
    age: 23,
//키에 값만 저장이 아니라 함수도 저장가능
display: function() {
            console.log(`이름: ${this.name}, 나이: ${this.age}`);
            // this는 객체 자신을 가리킨다.
            // 따라서 display() 메서드 내에서 this.name은 person.name과 같다.
            // this.생략불가, 람다로 대체불가
            // (람다=화살표 함수는 this를 바인딩하지 않음,못씀,사실상 함수표현식만 가능)

    },
    setValue:function(name, age) {
        this.name = name;
        this.age = age;
    }
};

person.display(); // 이름: 홍길동, 나이: 23
person.setValue("이순신", 30);
person.display(); // 이름: 이순신, 나이: 30

/*
객체는 키와 값의 쌍으로 이루어진 데이터 구조로,
키는 문자열로, 값은 다양한 데이터 타입(숫자, 문자열, 배열, 객체 등)을 가질 수 있다.
객체는 중괄호({})로 정의되며, 키와 값은 콜론(:)으로 구분된다.
객체는 프로퍼티(property)라고도 하며, 객체의 메서드는 함수로 정의되어 객체의 동작을 정의한다.
객체는 동적으로 프로퍼티를 추가하거나 수정할 수 있으며, 삭제할 수도 있다.
객체는 다른 객체를 포함할 수 있어 복잡한 데이터 구조를 표현할 수 있다.
객체는 JSON(JavaScript Object Notation) 형식으로 데이터를 표현할 수 있어, 데이터 전송 및 저장에 유용하다.
객체는 
//쌤PPT.javascript and query.pptx 

*/