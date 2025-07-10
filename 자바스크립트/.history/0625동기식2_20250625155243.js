//외부모듈끌고오기
let fs = require('fs');
let path = require('path');
try {
    data = fs.readFileSync('./0625동기식1.js', 'utf8');
    console.log(data);
} catch (err) {
    console.error(err);
}
console.log("파일읽기 끝");