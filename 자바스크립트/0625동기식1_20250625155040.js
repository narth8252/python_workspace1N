//속도빠름. 
function A() {
    let s = 0;
    for(i = 1; i <= 100; i++) {
        s += i;
    }
    console.log("합계", s);
}
function B() {
    let s = 0;
    for(i = 1; i <= 10; i++) {
        console.log(i);
    }
}

A();
B();