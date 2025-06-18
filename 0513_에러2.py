
print("---에러메세지표기 방법3.-----")
#점프투파이썬 05-4 예외처리 0513 2:24pm
#에러메시지 표기 try-except -finally구문

try:
    f = open("file1.txt", "r")
    lines = f.readlines()
    for line in lines:
        print(line)
except FileNotFoundError as e:  # 파일이 없을 경우 예외 처리
    print("파일을 찾을 수 없습니다:", e)

except FileExistsError as e:
    print(e)
finally:
    f.close()   # 파일이 없으면 f가 정의되지 않아서 여기서 에러 발생!

# finally:
#     try:
#         f.close()
#     except NameError:
#         pass  # f가 정의되지 않은 경우 (예: 파일이 없어서 open 실패)
"""
open("file1.txt", "r"): 파일 열기 (읽기 모드)
FileNotFoundError: 파일이 없을 때 발생하는 예외 (코드에서 기대하는 예외)
finally: 예외가 발생하든 안 하든 마지막에 실행됨 (파일 닫을 때 사용)
f.close(): 파일 자원 해제
"""
"""f.close()
    만약 f.close() 안 하면?
파일이 계속 열려 있어 메모리 낭비
여러 번 파일을 열었다가 안 닫으면 오류 발생 가능
다른 프로그램에서 그 파일을 편집하거나 삭제 못 할 수도 있음
    파일을 열면(예: open("파일명", "r"))
→ 컴퓨터는 해당 파일을 메모리에 올려서 작업합니다.
→ 작업이 끝났으면 꼭 닫아야 자원 낭비나 오류를 막을 수 있어요.
"""
"""
더 안전한 방법: with문 사용
with문을 쓰면 자동으로 닫아줘요!
"""



