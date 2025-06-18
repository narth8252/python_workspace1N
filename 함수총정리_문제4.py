def reverse(s):
    result = ""
    for i in range(len(s) -1, -1, -1):
        result += s[i]
    return result
print(reverse("Korea"))
