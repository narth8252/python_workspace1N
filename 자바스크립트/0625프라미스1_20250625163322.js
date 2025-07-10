/*
250625 pm4:20 ㅍ
프라미스(Promise) 객체는 비동기 작업의 완료 또는 실패를 나타내는 객체입니다.
프라미스는 비동기 작업의 결과를 나타내며, 성공적으로 완료되었을 때와 실패했을 때의 처리를 정의할 수 있습니다.
프라미스는 다음과 같은 세 가지 상태를 가집니다: 
1. 대기(pending): 프라미스가 아직 완료되지 않은 상태입니다.
2. 이행(fulfilled): 프라미스가 성공적으로 완료된 상태입니다.
3. 거부(rejected): 프라미스가 실패한 상태입니다.
프라미스는 비동기 작업을 처리하는 데 유용하며, 콜백 지옥(callback hell)을 피할 수 있는 방법 중 하나입니다.

*/
let promise = new Promise(function(resolve, reject) {
    // 비동기식으로 동작하는 함수
    sum = 0;
    for (let i = 0; i < 10; i++){
        sum += i;
    }
        // 성공적으로 작업을 완료했을 때
        resolve(sum); //return사용불가
        // reject("failed"); //실패했을 때는 catch구문으로 이동.
});  //리턴받는건 값55가 아니고 promise 객체.

//동기식을 비동기식으로 바꾸는 함수가 promise 객체입니다.
promise
.then((response) => {
    // 성공적으로 작업을 완료했을 때 실행되는 코드)     
    console.log(response);
    response = response*100;
    return response; //return은 then구문에서만 사용가능.
}).then((response) => {
    // 이전 then에서 리턴된 값을 받아서 실행되는 코드
    console.log("프라미스체인", response);
}).catch((error) => {
    // 작업이 실패했을 때 실행되는 코드
    console.log("작업이 실패했습니다. 에러:", error);
})
.finally(() => {
    // 성공 또는 실패 후에 항상 실행되는 코드
    console.log("작업이 완료되었습니다.");
});
console.log("프라미스가 생성되었습니다.");

function sigma(limil)