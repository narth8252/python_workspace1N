import { useState } from 'react'

function Counter() {
    const [count, setCount] = useState(0)

    return (
        <>
            <h1>카운터</h1>
            <h2>{count}</h2>
            <button type="hutton" onClick={()=>{ setCount(count+1)}}>증가</button>
            <button type="button" onClick={()=>{ setCount(count-1)}}>감소</button>
        </>
    )
}
export default Counter;
//파일이 다르면, 서로 못주고 받아서 파일내의 함수나 클래스를 외부로 노출시켜야
//다른 파일에서 이걸 사용가능

// cmd에서 서버작동해야 돌아감
// npm run dev
// cmd에 표기된 주소로 들어가 ➜  Local:   http://localhost:5173/


// # node_modules 폴더는 용량크니까 계속 지우고
// # pakage.json파일이 가장중요함

// 챕터 1: React 개발 환경 켜기
// cd C:\Users\Admin\Documents\GitHub\python_workspace1N\Frontend\React\react_project
// cd myhome_vite : 방금 만든 프로젝트 폴더로 이동합니다.
// npm run dev : 개발용 서버를 실행합니다. 
// 서버가 켜지면 터미널에 ➜ Local: http://localhost:5173 같은 주소가 표시됩니다. 
// 이 주소를 웹 브라우저에 입력하면 현재 프로젝트 화면을 볼 수 있습니다.

// 챕터 3: React의 핵심 개념 배우기
// 이제 프로젝트가 실행되는 상태에서 React의 핵심 개념인 컴포넌트와 **상태(State)**를 다룹니다.

// 1. 컴포넌트란?
// 컴포넌트는 웹사이트를 이루는 독립적인 UI 조각입니다. 재사용이 가능한 부품이라고 생각하면 이해하기 쉽습니다.
// 예시: 웹사이트의 Header, Footer, Sidebar 등은 모두 각각의 컴포넌트가 될 수 있습니다. 우리는 **Counter**라는 간단한 컴포넌트를 만들었습니다.

// 2. 상태(State)란?
// 컴포넌트가 가지고 있는 변화 가능한 데이터를 의미합니다. useState라는 훅을 사용해 상태를 선언하고 관리합니다.
// 예시: const [count, setCount] = useState(0)
// count: 현재 상태 값입니다.
// setCount: count 값을 변경하는 함수입니다. 이 함수를 사용해야만 화면이 자동으로 업데이트됩니다.

// 3. 컴포넌트 간의 관계
// React는 부모-자식 관계로 컴포넌트를 구성합니다.
// App.jsx: 부모 컴포넌트 역할을 합니다.
// Counter.jsx: 자식 컴포넌트 역할을 합니다.
// App 컴포넌트에서 <Counter /> 태그를 사용해 Counter 컴포넌트를 불러와서 화면에 표시합니다. 이렇게 하면 코드를 더 깔끔하게 관리할 수 있습니다.

// https://jsonplaceholder.typicode.com/todos