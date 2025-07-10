import re

text1 = "I like star"
text2 = "starship is beautiful"

pattern = "star"
print (re.match( pattern, text1)) #None
print (re.match( pattern, text2)) #

matchObj = re.match( pattern, text2)
print(matchObj.group() )
print(matchObj.start() )
print(matchObj.end() )
print(matchObj.span() )
print( text2[:4])
