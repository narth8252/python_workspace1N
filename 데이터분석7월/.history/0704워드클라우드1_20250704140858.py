# pip install simplejson
import simplejson
import webbrowser
import pytagcloud

tag = [
    ('school', 30),
    ('rainbow', 70),
    ('NJS', 300),
    ('BlackPink', 39),
    ('HyoRi', 70),
    ('IVE', 210),
    ('game', 350),
    ('oneyoung',20),
    ('twice',50),
    ('HOT',100),
    ('SES',500)
]
taglist = pytagcloud.make_tags(tag, maxsize=50)
print(taglist)
pytagcloud.create