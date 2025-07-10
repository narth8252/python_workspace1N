/*
0625 PM4시 비동기식
함수가 시작하자마자 바로 리턴
시스템이 백그라운드에서 조용히 일함
시스템입장에서는 일이끝나면 알려준다

*/
let fs = require('fs');

fs.readFile('example.txt', 'utf8', (err, data) => {
  if (err) {
    console.error(err);
    return;
  }
  console.log(data);
});