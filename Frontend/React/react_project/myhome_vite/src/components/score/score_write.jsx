import { useState, useEffect } from "react";
import axios from 'axios';
import { Link, useNavigate  } from "react-router-dom";

function ScoreWrite(){
    const [score, setScore] = useState({name:"", kor:0, eng:0, mat:0});
    let history = useNavigate ();    

    //해체(모던스크립트 문법) score -> name,kor,eng
    const {name, kor, eng, mat } = score; 

    //onChange 함수 개별로 붙이기 귀찮아서 
    const onChange= (e)=>{
        const {value, name} = e.target; //value, name 
        //value = e.target.value;
        //name = e.target.name; 각각쓰는걸 위로 합침 (모던스크립트 문법)
        setScore({...score, [name]:value});  
    }

    const onSubmit=(e)=>{
        e.preventDefault();  //서버로 전송되는 원래 기능을 막아야하 한다. 
 
        // axios.post("http://127.0.0.1:8000/score/insert", score)
        axios.post("http://127.0.0.1:8000/score/score/insert", score)
        .then((res)=>{
            alert("등록성공");
            history('/score/list');
            console.log("등록성공");
            //navigate 사용해서 이동시켜야함
            //앵커태그,locationhref 사용불가)->page자체를 변경시키므로 쓰지마
        })
        .catch((error)=>{
            console.log(error);
        })
    }
    return(
        <div>
            <form onSubmit={onSubmit}>
            이름 : <input type="text" val={name} name="name" onChange={onChange}/> <br/>
            국어 : <input type="text" val={kor}  name="kor" onChange={onChange}/> <br/>
            영어 : <input type="text" val={eng}  name="eng" onChange={onChange}/> <br/>
            수학 : <input type="text" val={mat}  name="mat" onChange={onChange}/> <br/>
            <button type="submit">등록</button>
            </form>
            
        </div>    
         
    ) 
    
}

export default ScoreWrite;

// 코드 해설
// useState: score 객체를 상태로 관리합니다. name, kor, eng, mat 필드의 초기값을 설정했습니다.
// handleChange 함수: 이 함수는 <input> 필드의 값이 변경될 때마다 호출됩니다.
    // e.target을 사용해 이벤트가 발생한 <input> 태그의 name 속성과 value 속성을 가져옵니다.
    // setScore를 호출하여 현재 score 상태를 복사(...score)한 뒤, 변경된 필드의 값([name]: value)만 업데이트합니다.
// handleInsert 함수: '추가' 버튼을 클릭했을 때 실행될 함수입니다. 이 함수 안에 axios를 사용해 백엔드 서버로 데이터를 전송하는 로직을 추가하면 됩니다.
// 이 코드는 사용자의 입력에 따라 상태가 실시간으로 변하는, React의 가장 기본적인 상호작용 방식을 보여줍니다. 
// 이제 이 컴포넌트를 사용해 성적을 입력하고, '추가' 버튼을 눌렀을 때 백엔드로 데이터를 보내는 다음 단계를 진행할 수 있습니다.