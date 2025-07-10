/*
0625 PM4시 비동기식
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

