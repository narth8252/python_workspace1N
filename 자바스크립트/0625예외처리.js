// let jsondata = "{bad json}"; //잘못된 JSON 형식
//객체리터럴은 '' ""없거나 다되는데, json은 키값에 반드시 ""만된다.
//객체리터럴은 키값에 '' ""없거나 둘다 가능하다.
let jsondata = '{"name": "홍길동", "age": 23}'; //올바른 JSON 형식

try {
    //데이터송수신할때 실제로는 json객체를 주고받는게 아니고
    //json형태의 문자열을 주고받는다. 그래서 파싱작업을 해야한다.
    //JSON.parse() 메서드를 사용하여 JSON 문자열을 JavaScript 객체로 변환
    //             JSON 문자열이 올바른 형식이어야 하며, 그렇지 않으면 오류가 발생한다.
    //↔ JSON.stringify() 메서드는 JavaScript 객체를 JSON 문자열로 변환하는 데 사용된다.
    
    let user = JSON.parse(jsondata); // JSON 파싱 시도
    console.log(user.name, user.age); // "홍길동", 23
} catch (error) {
    console.error("JSON 파싱 오류:", error);
}