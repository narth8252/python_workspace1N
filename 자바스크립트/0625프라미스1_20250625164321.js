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

//일반함수fuction앞에 어싱크를 붙이면 
async function sigma(limit=10){
    let s = 0;
    for(let i = 1; i <= limit; i++){
        s += i;
    }
    return s;
}
sigma(100).then(console.log); // 또는 async 함수 안에서 await sigma(100)
/*
설명
ㆍsync 함수는 항상 Promise를 반환합니다.
ㆍ값을 바로 출력하려면 .then() 또는 await를 사용해야 실제 결과를 볼 수 있습니다.
정리
ㆍPromise는 비동기 작업의 완료/실패를 나타내는 객체로, pending, fulfilled, rejected 세 가지 상태를 가집니다.
ㆍ.then(), .catch(), .finally()를 통해 결과 처리 및 에러 처리가 가능합니다.
ㆍasync 함수는 항상 Promise를 반환하며, 값을 얻으려면 .then()이나 await를 사용해야 합니다.
ㆍ코드의 변수명, for문 조건 등 오타를 주의해야 하며, Promise와 async/await의 동작 원리를 이해하면 비동기 코드를 효율적으로 작성할 수 있습니다.
*/

//어떤경우 비동기로 꼭 처리해야할때 있는데
/*
    비동기
            비동기
                    비동기
                            비동기 - 멸망의피라미드
    await: 프라미스 객체가 일을 마무리하길 기다린다
    await명령어가 async를 기다리는데 이 구문 자체가 async함수에서만 사용가능
    아래예제에서 mainModu
*/
async function mainModule(){
    let result = await sigma(100);
    console.log("결과", result);
    console.log("ending .....");
}
mainModule();

