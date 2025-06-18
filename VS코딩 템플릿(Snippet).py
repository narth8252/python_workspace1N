# 탭이나 컨트롤키 등으로 커서를 자동완성된 다음으로 넘길순없어?
# 좋은 질문이야! 자동 완성된 닫힌 괄호나 따옴표 뒤로 커서를 빠르게 넘기는 데 **탭(Tab)** 또는 **컨트롤(Control)** 키를 활용할 수 있는지에 대해 살펴보자.
"""
## 1. **탭(Tab) 키로 넘기기?**
* 기본적으로 대부분 에디터는 **탭 키가 들여쓰기(Indent) 용도**라서, 닫힌 괄호 뒤로 커서 넘기는 용도로는 쓰지 않아.
* 일부 에디터나 확장 프로그램에서 커서 자동 이동 기능을 탭 키에 할당할 수 있지만, 기본 설정은 아님.
* 예를 들어 VS Code에선 [Tab Out](https://marketplace.visualstudio.com/items?itemName=albert.TabOut) 같은 확장 프로그램을 설치하면
  * 자동 완성된 괄호, 따옴표, 대괄호 뒤로 **탭키 한 번으로 커서 이동 가능**해진다.

## 2. **컨트롤(Control) 키 조합으로 넘기기?**
* 기본 에디터 단축키로 컨트롤과 방향키 조합(예: `Ctrl + →`)는 단어 단위로 커서 이동한다.
* 닫힌 괄호나 따옴표 단위로만 넘기진 않고, 문장 전체에서 단어별 이동이라 좀 과할 수 있음.
* 자동 완성 후 바로 닫힌 부분으로 커서 이동하는 기능과는 별개임.

## 3. **요약 및 추천**
| 방법                    | 가능 여부/특징                                |
| ----------------------- | -------------------------------------------- |
| 탭키 기본 기능           | 들여쓰기, 자동 완성 닫힌 괄호 이동 아님         |
| 탭키로 커서 이동 기능 확장| VS Code ‘Tab Out’ 같은 확장 설치해야 가능      |
| Ctrl + 방향키           | 단어 단위 커서 이동, 자동완성 닫힌 괄호 이동 아님|
| 닫힌 괄호 키 누르기      | 기본적으로 커서 자동 이동 기능, 가장 빠름       |

## 4. **VS Code에서 Tab Out 확장 설치**
1. VS Code 좌측 확장(Extensions) 메뉴 클릭
2. `Tab Out` 검색 후 설치
3. 자동 완성된 괄호, 따옴표 뒤에서 `Tab` 키 누르면 커서 바로 이동

필요하면 네가 쓰는 에디터 환경 알려줘. 그 환경에 맞게 더 딱 맞는 단축키나 플러그인 방법 알려줄게.
자동완성 커서 이동, 작업 효율 엄청 올려줄 수 있는 부분이라 핵심이야!

✅ TabOut이 탭키에 적용하는 기본 규칙
커서가 괄호, 따옴표 안에 있을 때
→ Tab 키는 커서를 닫힌 괄호나 따옴표 바깥으로 이동시킴.
예: (|) 상태에서 Tab 누르면 ()로 커서가 나감.

그 외 일반 상황(코드 중간, 라인 시작 등)
→ Tab 키는 **원래처럼 들여쓰기(indent)**로 작동함.
예: 줄 맨 앞에서 Tab 누르면 그대로 스페이스 4칸(혹은 설정한 만큼) 들어감.
"""

# Tip: 설정을 커스터마이징하고 싶다면
# settings.json에 아래 항목을 추가하거나 확인해볼 수 있어
# "activationKeys": 어떤 키로 커서 이동시킬지 설정 (Tab, Enter 등 가능)
# "pairs": 어떤 쌍 문자에서 동작할지 설정
"tabout.enable": true,
"tabout.activationKeys": ["tab"],
"tabout.pairs": ["()", "[]", "{}", "''", "\"\""]

"""
미세한 자동완성 커서 이동 기능
특히 VS Code에서 쓰는 사람이라면 반드시 알고 있으면 좋을 기능과 설정들이야.

## 1. TabOut 플러그인 활용(기본 자동 괄호 이동 강화)
 `Tab` 키로 자동완성 괄호/따옴표 밖으로 커서 이동
 커서가 괄호 앞에 있지 않으면 원래대로 들여쓰기
 지원하는 쌍: `()`, `{}`, `[]`, `''`, `""` 등

## 2. 자동완성 Snippet에서 커서 점프하기(정말 유용)
코드를 자동완성할때 여러위치로 커서를 이동시켜야 하는경우, 
snippet에 $1, $2 같은 커서점프포인트(tab stop)를 써서 탭키로 점프
"""
###예시:json
"Print with variable": {
  "prefix": "pvar",
  "body": [
    "print(\"$1 =\", $1)"
  ],
  "description": "Print variable with label"
}

# pvar 입력후 Tab 누르면 print("커서 = ", 커서) 자동입력
# 첫번째 $1에 커서가 가고, 다시 Tab누르면 두 번째 $1로 이동
#복잡한 코드 템플릿 반복할 때 매우 유용
"""
## 3. Surround 기능으로 괄호 자동감싸기 (Wrap with Bracket)
텍스트 선택 후 '(' 혹은 '{' 입력하면 자동으로 (선택내용)로 감쌈
커서는 닫힌 괄호 바깥에 놓임

```plaintext
선택된 텍스트: print("hello")
입력: (
결과: (print("hello"))
```
"""
# 설정에서 아래 항목 활성화 필요:json
"editor.autoSurround": "languageDefined"

"""
##4. **VS Code 단축키 활용**
| 단축키             | 기능                       |
| ------------------ | ------------------------- |
| Ctrl + → / ←       | 단어단위로 커서 이동          |
| Ctrl + Shift + →   | 단어단위로 블록 선택          |
| Ctrl + Backspace   | 단어단위 삭제                 |
| Ctrl + D           | 같은단어 한번더 선택(멀티커서) |
| Ctrl + Shift + L   | 선택된단어 전체에 멀티커서 생성|
|Alt + ↑ / ↓         | 줄 이동                      |
|Shift + Alt + → / ← | 코드 블록단위 선택(괄호기준 자동선택)|
"""
## 5. **Editor Config 설정 예시 json
"editor.autoClosingBrackets": "languageDefined",  
"editor.autoClosingQuotes": "languageDefined",  
"editor.tabCompletion": "on",  
"editor.snippetSuggestions": "top",  
"editor.autoSurround": "languageDefined"
"""
## 결론
| 기능             | 설명 및 키워드                 |
| ---------------- | ----------------------------- |
| TabOut           | 자동완성 괄호바깥으로 Tab이동   |
| Snippets($1,$2)  | 템플릿내부 커서점프             |
| Auto Surround    | 괄호로감싸기(autoSurround)      |
| 커서이동 단축키    | Ctrl,Alt,Shift조합으로 미세제어 |
| 멀티커서/다중선택  | Ctrl+D, Ctrl+Shift+L 등  |
"""
# VS Code에서 직접만든 자동화 코딩템플릿(Snippet): TabOut+Snippet+Surround 조합
# 설정하면 VS Code가 꺼져도, 재부팅해도, 프로젝트를 바꿔도 계속 사용 가능해.
"""
##  Snippet(코딩 템플릿) 적용 방식
### 방법 1: **글로벌 사용자 스니펫**
1. **단축키 `Ctrl + Shift + P` → "Preferences: Configure User Snippets"**
2. `New Global Snippets file` 선택
3. 예: `my-snippets.code-snippets` 이름으로 생성
4. 아래처럼 템플릿 작성 👇
"""
```json
{
  "Print Variable with Label": {
    "prefix": "pvar",
    "body": [
      "print(\"$1 =\", $1)"
    ],
    "description": "Print variable with label"
  },
  "Function Skeleton": {
    "prefix": "defx",
    "body": [
      "def $1($2):",
      "    $3",
      ""
    ],
    "description": "Function skeleton with parameters"
  }
}

"""
##  사용법
 코드창에서 pvar 쓰고 Tab 누르면
  👉 print("커서 = ", 커서) 자동완성+커서이동
defx 쓰고 Tab 누르면
  👉 def ( ): 틀이 뜨고 커서가 매개변수와 함수본문으로 점프가능
"""
##  추천 Snippet 패턴예시
### 1.자주 쓰는 함수 작성용 json
"Print debug": {
  "prefix": "dbg",
  "body": ["print(f'DEBUG: $1')"],
  "description": "Quick debug print"
}

### 2.파일 입출력 json
"Read file line by line": {
  "prefix": "rfile",
  "body": [
    "with open('$1', 'r') as f:",
    "    lines = f.readlines()",
    "    for line in lines:",
    "        print(line.strip())"
  ],
  "description": "Read file lines"
}

### 3.조건문 블럭 json
"If block": {
  "prefix": "ifx",
  "body": [
    "if $1:",
    "    $2"
  ],
  "description": "If statement block"
}

"""
## 📁 Snippets 파일 저장 경로:
Windows: C:\Users\<사용자이름>\AppData\Roaming\Code\User\snippets\
VS Code에서 자동으로 관리되니 걱정할 건 없음. 파일삭제하지 않는한 영구유지됨.

## 정리
| 한 번 설정하면 계속? | 유지됨. 한번만 설정하면 됨          |
| 어디서든 써도 되나?  | 글로벌 스니펫이면 모든 프로젝트에서 사용가능 |
| 커서 이동도 가능?    | $1, $2 등으로 탭누르면 점프됨  |
| 커스터마이징 가능?   | 이름,단축어,내용 자유롭게 조절가능    |

필요하면 `너만의 스니펫 팩` 만들어서 하나로 묶어줄게.
자주쓰는 코드패턴(예: 함수, print, 조건문, 반복문, 파일처리, 예외처리 등) 알려줘봐
전부 자동화해줄 수 있어.
"""
"""글로벌snippet.code-snippets파일에 전체붙여넣기
C:\Users\Administrator\AppData\Roaming\Code\User\snippets\narsnippet.code-snippets
#  VS Code: Ctrl + Shift + P → Preferences: Configure User Snippets → New Global Snippets file
# 예: python-kit.code-snippets → 붙여넣고 저장
{
	"Print Variable": {
	  "prefix": "pvar",
	  "body": [
		  "print(\"$1 =\", $1)"
	  ],
	  "description": "Print variable name and value"
	},
	"Function Skeleton": {
	  "prefix": "defx",
	  "body": [
      "def $1($2):",
      "    $3"
	  ],
	  "description": "Function skeleton"
	},
	"For Loop": {
	  "prefix": "forx",
	  "body": [
      "for $1 in $2:",
      "    $3"
	  ],
	  "description": "Basic for loop"
	},
	"If Block": {
	  "prefix": "ifx",
	  "body": [
      "if $1:",
      "    $2"
	  ],
	  "description": "Basic if block"
	},
	"Elif Else Block": {
	  "prefix": "elifx",
	  "body": [
      "if $1:",
      "    $2",
      "elif $3:",
      "    $4",
      "else:",
      "    $5"
	  ],
	  "description": "If-Elif-Else block"
	},
	"Try Except": {
	  "prefix": "tryx",
	  "body": [
      "try:",
      "    $1",
      "except $2 as e:",
      "    print(e)"
	  ],
	  "description": "Try-except block"
	},
	"List Comprehension": {
	  "prefix": "listx",
    "body": [
      "[$1 for $2 in $3 if $4]"
	  ],
	  "description": "List comprehension template"
	},
	"Read File Line By Line": {
	  "prefix": "rfile",
	  "body": [
      "with open(\"$1\", \"r\") as f:",
      "    for line in f:",
      "        print(line.strip())"
	  ],
	  "description": "Read file line by line"
	},
	"Lambda Function": {
	  "prefix": "lambx",
	  "body": [
		  "lambda $1: $2"
	  ],
	  "description": "Lambda function"
	},
	"Sort With Key": {
	  "prefix": "sortx",
	  "body": [
		  "$1.sort(key=lambda x: x[$2], reverse=$3)"
    ],
	  "description": "List sort with key and reverse"
	},
	"Dictionary Iteration": {
	  "prefix": "dictx",
	  "body": [
      "for key, value in $1.items():",
      "    print(f\"{key}: {value}\")"
	  ],
	  "description": "Iterate over dictionary"
	},
	"Input Processing": {
	  "prefix": "inputx",
	  "body": [
      "$1 = input(\"$2: \")",
      "# split example:",
      "$3 = input().split()",
      "# map(int, input().split()) example:",
      "$4 = list(map(int, input().split()))"
	  ],
	  "description": "Various input() processing patterns"
	},
	"Datetime Now": {
	  "prefix": "dtnow",
	  "body": [
		  "from datetime import datetime",
		  "now = datetime.now()",
		  "print(now.strftime(\"%Y-%m-%d %H:%M:%S\"))"
	  ],
	  "description": "Get current date and time"
	},
	"Enumerate Loop": {
	  "prefix": "enumx",
	  "body": [
      "for idx, val in enumerate($1):",
      "    print(idx, val)"
	  ],
	  "description": "Loop with enumerate"
	},
	"Write File": {
	  "prefix": "wfile",
	  "body": [
		  "with open(\"$1\", \"w\") as f:",
		  "    f.write($2)"
	  ],
	  "description": "Write to file"
	},
	"Set From List": {
    "prefix": "setx",
    "body": [
      "$1_set = set($2)",
      "print($1_set)"
    ],
    "description": "Convert list to set"
  },
  "DefaultDict": {
    "prefix": "ddictx",
    "body": [
      "from collections import defaultdict",
      "$1 = defaultdict($2)",
      "$1[$3] += 1"
      ],
	    "description": "defaultdict usage example"
  	}
  }
  

# 사용 예시
# pvar + [Tab]  →  print("변수이름 =", 변수이름) 자동완성
# defx + [Tab]  →  def 함수명(매개변수):
#                    본문
# 👉 탭 누르면 $1, $2, $3 위치로 커서 자동 점프됨.

| prefix         | 내용 설명                      |
| -------------- | -------------------------- |
| `pvar`         | 변수명과 값 함께 출력               |
| `defx`         | 함수 템플릿                     |
| `forx`         | 기본 for 루프                  |
| `ifx`, `elifx` | 조건문 패턴                     |
| `tryx`         | try-except 구조              |
| `listx`        | 리스트 컴프리헨션                  |
| `rfile`        | 파일 한 줄씩 읽기 (`read`)        |
| `wfile`        | 파일 쓰기 (`write`)            |
| `dictx`        | 딕셔너리 순회                    |
| `inputx`       | input(), split(), map 처리   |
| `sortx`        | 리스트 정렬 (key, reverse 포함)   |
| `lambx`        | 람다 함수                      |
| `dtnow`        | 현재 날짜시간 구하기                |
| `enumx`        | enumerate() 루프             |
| `setx`         | 리스트를 set으로 변환              |
| `ddictx`       | collections.defaultdict 사용 |
 
"""
"""settings.json파일에 전체붙여넣기 
C:\Users\Administrator\AppData\Roaming\Code\User\settings.json
# Ctrl + Shift + P (Cmd + Shift + P on Mac) 눌러서 명령 팔레트 실행
# settings 입력
# “Preferences: Open User Settings (JSON)” 선택
# 메뉴→파일>기본설정>설정 또는 Ctrl+, 단축키
# 오른쪽 상단 {} 아이콘 클릭 → settings.json 편집기 열림
# → 사용자 전역 설정용 settings.json 열림

{
  "security.workspace.trust.untrustedFiles": "open",
  "workbench.colorTheme": "Visual Studio Light",
  "files.autoSave": "afterDelay",
  "python.defaultInterpreterPath": "C:\\ProgramData\\anaconda3\\python.exe",
  "editor.fontSize": 16,
  "editor.defaultFormatter": "ms-python.black-formatter",
  "editor.formatOnSave": true,
  "pylint.enabled": false,

  "tabout.activationOnEnter": false,
  "tabout.activationOnTab": true,
  "tabout.pairs": [
      ["\"", "\""],
      ["'", "'"],
      ["(", ")"],
      ["[", "]"],
      ["{", "}"]
  ],

  "python.languageServer": "Pylance",
  "python.formatting.provider": "black",
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  "files.autoSaveDelay": 1000,

  "editor.autoClosingBrackets": "beforeWhitespace",
  "editor.autoClosingQuotes": "always",

  "editor.lineNumbers": "on"
}

"""
"""컬러랑 폰트 프로그래머스처럼 바꾼 {}settings.json
{
  "security.workspace.trust.untrustedFiles": "open",
  "workbench.colorTheme": "Visual Studio Light",
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,

  "python.defaultInterpreterPath": "C:\\ProgramData\\anaconda3\\python.exe",
  "python.languageServer": "Pylance",
  "python.formatting.provider": "black",

  "editor.defaultFormatter": "ms-python.black-formatter",
  "editor.formatOnSave": true,
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  "editor.fontSize": 16,
  "editor.fontFamily": "Fira Code, Consolas, 'Courier New', monospace",
  "editor.fontLigatures": true,
  "editor.lineNumbers": "on",

  "editor.autoClosingBrackets": "always",
  "editor.autoClosingQuotes": "always",

  "pylint.enabled": false,

  "tabout.activationOnEnter": false,
  "tabout.activationOnTab": true,
  "tabout.pairs": [
    ["\"", "\""],
    ["'", "'"],
    ["(", ")"],
    ["[", "]"],
    ["{", "}"]
  ],

  "editor.tokenColorCustomizations": {
    "textMateRules": [
      {
        "scope": "keyword",
        "settings": { "foreground": "#c30771" }
      },
      {
        "scope": "variable",
        "settings": { "foreground": "#008ec4" }
      },
      {
        "scope": "string",
        "settings": { "foreground": "#10a778" }
      },
      {
        "scope": "constant.numeric",
        "settings": { "foreground": "#ff8800" }
      },
      {
        "scope": "comment",
        "settings": { "foreground": "#999999", "fontStyle": "italic" }
      },
      {
        "scope": "entity.name.function",
        "settings": { "foreground": "#7a5ccc" }
      }
    ]
  }
}

"""

"""
##1. 인기테마설치(프로그래머스 느낌과 비슷한추천)
아래 중 하나 설치하고 적용
| 테마 이름           | 설명                                       |
| ------------------ | ------------------------------------------ |
| One Dark Pro       | 가장인기많고 가독성좋은테마. 프로그래머스유사한 차분한대비색상 |
| Tokyo Night        | 파란계열 메인, 부드럽고 현대적인 느낌.                |
| Dracula Official   | 어두운배경에 고채도색상. 시각적 분명함.                |
| GitHub Dark / Light| 웹 IDE 느낌. GitHub에서 코딩하는 느낌.                |
| Panda Theme        | 귀엽고 화사한 파스텔계열. 집중용으로도 적절.           |

### 설치 방법:
1. VSCode 왼쪽사이드바 → 확장(Extensions) (Ctrl + Shift + X)
2. 테마 이름 검색 (예: One Dark Pro)
3. 설치+적용

##  2. 폰트도 같이 세팅하면 완성도 ↑
### 추천 개발용 폰트:
Fira Code (→ ligature 기능으로 => 등 예쁘게 보임)
JetBrains Mono
Cascadia Code

### 설정추가 settings.json
"editor.fontFamily": "Fira Code, Consolas, 'Courier New', monospace",
"editor.fontLigatures": true,
"editor.fontSize": 16

## 3.아이콘테마까지 변경 (전체분위기통일)
Material Icon Theme
  → 폴더/파일에 맞는 직관적 아이콘 제공
설치방법은 테마동일: 확장탭 → Material Icon Theme 검색 → 설치 → 적용

## 4. UI 배경도 바꾸고 싶다면?
확장프로그램 Background 사용해서 배경이미지 넣을수도.(무거워질수있음)

## 추천조합(프로그래머스 느낌)
* 테마: One Dark Pro
* 폰트: Fira Code + Ligature on
* 아이콘 테마: Material Icon Theme
* 추가설정: "editor.fontSize": 16, "editor.lineHeight": 24, "editor.cursorSmoothCaretAnimation": true

필요하면 내가 이 조합을 `settings.json` 형식으로 완성해줄 수도 있어.
원하는 스타일 (밝은 / 어두운 / 차분함 / 고대비 등) 말해주면 더 맞춤 추천 가능.

## 목표
 VS Code 전체 UI**: 밝은(라이트) 테마
코드색상: 프로그래머스 다크모드에서 보이는 것처럼 직관적인 문법색상(파랑, 주황, 보라, 초록 등)
폰트: Fira Code로 세련된 개발용 글꼴


## 추천 테마 조합 (라이트 UI + 다크 색상)
| 항목                 | 추천                                             |
| -------------------- | ------------------------------------------------ |
| UI 테마              | Visual Studio Light (기본테마, 깔끔)               |
| 문법색상만 변경할테마 | Light(default light)+[Customized Semantic Colors] |
| 폰트                 | Fira Code + Ligature 사용                       
## 설정방법 요약
### 1. VS Code 라이트테마설정
 Command Palette 열기: Ctrl + Shift + P
 Color Theme 입력 → Light+ (default light) 또는 Visual Studio Light 선택

### 2. Fira Code 폰트설치 및 설정
```json
"editor.fontFamily": "Fira Code, Consolas, 'Courier New', monospace",
"editor.fontLigatures": true,
"editor.fontSize": 16
```

### 3. 문법색상만 커스터마이징
workbench.colorCustomizations와 editor.tokenColorCustomizations 테마색상커스터마이징.
#### settings.json`에 추가
"editor.tokenColorCustomizations": {
  "textMateRules": [
    {
      "scope": "keyword",
      "settings": { "foreground": "#c30771" }  // 빨간 키워드 (def, return 등)
    },
    {
      "scope": "variable",
      "settings": { "foreground": "#008ec4" }  // 파랑 변수
    },
    {
      "scope": "string",
      "settings": { "foreground": "#10a778" }  // 초록 문자열
    },
    {
      "scope": "constant.numeric",
      "settings": { "foreground": "#ff8800" }  // 주황 숫자
    },
    {
      "scope": "comment",
      "settings": { "foreground": "#999999", "fontStyle": "italic" }
    },
    {
      "scope": "entity.name.function",
      "settings": { "foreground": "#7a5ccc" }  // 보라 함수
    }
  ]
}
```
##  설정 파일 예시 (통합)json
{
  "security.workspace.trust.untrustedFiles": "open",
  "workbench.colorTheme": "Visual Studio Light",
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,
  "python.defaultInterpreterPath": "C:\\ProgramData\\anaconda3\\python.exe",
  "editor.fontSize": 16,
  "editor.fontFamily": "Fira Code, Consolas, 'Courier New', monospace",
  "editor.fontLigatures": true,
  "editor.defaultFormatter": "ms-python.black-formatter",
  "editor.formatOnSave": true,
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  "editor.lineNumbers": "on",
  "editor.autoClosingBrackets": "always",
  "editor.autoClosingQuotes": "always",
  "pylint.enabled": false,
  "python.languageServer": "Pylance",
  "python.formatting.provider": "black",

  // TabOut 확장 설정
  "tabout.activationOnEnter": false,
  "tabout.activationOnTab": true,
  "tabout.pairs": [
    ["\"", "\""],
    ["'", "'"],
    ["(", ")"],
    ["[", "]"],
    ["{", "}"]
  ],

  // 커스텀 코드 색상
  "editor.tokenColorCustomizations": {
    "textMateRules": [
      {
        "scope": "source",
        "settings": { "foreground": "#16ca05" }  // 기본 코드
      },
      {
        "scope": "keyword",
        "settings": { "foreground": "#f8038e" }  // 키워드
      },
      {
        "scope": "variable",
        "settings": { "foreground": "#b90b0b" }  // 변수명
      },
      {
        "scope": "string.quoted.double.python",
        "settings": { "foreground": "#f7cb09" }  // "문자열"
      },
      {
        "scope": "string.quoted.single.python",
        "settings": { "foreground": "#f88705" }  // '문자열'
      },
      {
        "scope": "string.quoted.docstring.multi.python",
        "settings": { "foreground": "#d41a1ac7" }  // 멀티라인 """문자열"""
      },
      {
        "scope": "constant.numeric",
        "settings": { "foreground": "#e10cfdfb" }  // 숫자
      },
      {
        "scope": "comment",
        "settings": {
          "foreground": "#5858589f",
          "fontStyle": "italic"  // 이탤릭 있음
        }
      },
      {
        "scope": "entity.name.function",
        "settings": { "foreground": "#233bc7f1" }  // 함수 이름
      }
    ]
  }
}

필요하면 내가 `.json` 파일 형태로 내보내는 법, 적용 위치, 폰트 설치 링크까지도 도와줄게.
더 강조하고 싶은 문법 요소(예: 클래스, 연산자 등) 있으면 말해줘서 맞춤 색상도 지정 가능.

"""