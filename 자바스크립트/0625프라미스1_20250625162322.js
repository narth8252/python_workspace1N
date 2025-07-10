/*
250625 pm
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