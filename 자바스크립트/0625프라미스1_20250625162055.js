let promise = new Promise(function(resolve, reject) {
    // 비동기식으로 동작하는 함수
    setTimeout(function() {
        // 성공적으로 작업을 완료했을 때
        resolve("작업이 성공적으로 완료되었습니다.");
    }, 2000); // 2초 후에 실행
}