#쌤ppt-19~20p 0514 11:30am
#◈ sub 함수 예제파일명 : exam13_6.py
#함수가 중요한게 아니라 pattern만드는게 중요

import re

text1 = " I Like stars, red star, yellow star"

print()
pattern = "star"
result = re.sub( pattern, "moon", text1)
print(result) # I Like moons, red moon, yellow moon

result2 = re.sub( pattern, "moon", text1, count = 2)
print(result2) # I Like moons, red moon, yellow star

# I Like moons, red moon, yellow moon
# I Like moons, red moon, yellow star