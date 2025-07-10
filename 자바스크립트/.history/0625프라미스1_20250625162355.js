/*
250625 pm4:20 ㅍ
프라미스(Promise) 객체는 비동기 작업의 완료 또는 실패를 나타내는 객체입니다.
프라미스는 비동기 작업의 결과를 나타내며, 성공적으로 완료되었을 때와 실패했을 때의 처리를 정의할 수 있습니다.
프라미스는 다음과 같은 세 가지 상태를 가집니다: 
1. 대기(pending): 프라미스가 아직 완료되지 않은 상태입니다.
2. 이행(fulfilled): 프라미스가 성공적으로 완료된 상태입니다.
3. 거부(rejected): 프라미스가 실패한 상태입니다.
프라미스는 비동기 작업을 처리하는 데 유용하며, 콜백 지옥(callback hell)을 피할 수 있는 방법 중 하나입니다.
// 프라미스(Promise) 객체 생성 예제
*/
let promise = new Promise(function(resolve, reject) {
    // 비동기식으로 동작하는 함수
    sum = 0;
    for (let i = 0; i < 10; i++){
        sum += i;
    }
        // 성공적으로 작업을 완료했을 때
        resolve(sum); //return사용불가
});
console.log("Promise 생성됨");

let promise2 = new Promise(function(resolve, reject) {
    // 비동기식으로 동작하는 함수
    setTimeout(function() {
        // 실패했을 때
        reject("작업이 실패했습니다.");
    }, 1000); // 1초 후에 실행
});