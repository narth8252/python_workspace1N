//외부모듈끌고오기
let fs = require('fs');
//비동기식 파일읽기
try {
    data = fs.readFileSync('./0625동기식1.js', 'utf8');
    console.log(data);
} catch (err) {
    console.error(err);
}
console.log("파일읽기 끝");