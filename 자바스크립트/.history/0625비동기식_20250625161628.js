/*
0625 PM4시 비동기식 javascript and jquery백쌤ppt-110p.
멸망의 피라미드
C:\Users\Admin\Documents\GitHub\python_workspace1N\자바스크립트
함수가 시작하자마자 바로 리턴
시스템이 백그라운드에서 조용히 일함
비동기식은 시스템이 일을 끝내면 알려준다.
자바스크립트는 콜백함수를 사용
함수를 내가만들고, 호출은 시스템이 한다.

*/
let fs = require('fs');

fs.readFile('./0625동기식1.js', 'utf8', function(err, data) {
  console.log(data);
});

console.log("파일읽기 끝");
// 비동기식은 시스템이 일을 끝내면 알려준다.