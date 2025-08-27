import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Counter from './components/counter';
import {Link, Routes, Route, Outlet} from "react-router-dom"
import "bootstrap/dist/css/bootstrap.min.css"

import Home from "./pages/home"; //확장자 생략
import About from "./pages/about";
import Nomatch from "./pages/nomatch";
import ScoreList from './components/score/score_list';
import ScoreWrite from './components/score/score_write';
import BoardList from './components/board/board_list';
import BoardWrite from './components/board/board_write';

function App() {
  const [count, setCount] = useState(0)
  
  return (
    <div className='container-fluid'>
      <nav style={{display:"flex", gap:"1rem", marginBottom:"1rem"}}>
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
        <Link to="/counter">Counter</Link>
        <Link to="/score/list">성적처리</Link>
        <Link to="/board/list">게시판</Link>
      </nav>

      {/* Routes - 경로  url ->특정컴포넌트와 연결하는 작업*/}
      <Routes>
        <Route path="/" element={<Home/>} />
        <Route path="/about" element={<About/>} />
        <Route path="/counter" element={<Counter/>} />
        <Route path="/score/list" element={<ScoreList/>} />
        <Route path="/score/insert" element={<ScoreWrite/>} />
        <Route path="/board/list"   element={<BoardList/>} />
        <Route path="/board/insert" element={<BoardWrite/>} />
        <Route path="*" element={<Nomatch/>} />
      </Routes>
      { /*url을 바꾸면 컴포넌트가 출력될 위치*/}


    
      <Outlet/>
    </div>
  )
}

export default App

// components폴더>파일들 수정➜여기에 <Link to랑 <Route 추가➜ npm run dev➜링크들어가
// cd C:\Users\Admin\Documents\GitHub\python_workspace1N\Frontend\React\react_project\myhome_vite
// cmd에서 서버작동해야 돌아감
// npm run dev
// cmd에 표기된 주소로 들어가 ➜  Local:   http://localhost:5173/
// http://localhost:5173/board/list
// http://127.0.0.1:8000/docs#/
// http://127.0.0.1:8000/board/list
