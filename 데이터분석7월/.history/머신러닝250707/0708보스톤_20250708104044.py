from sklearn.datasets import fetch_openml
boston = fetch_openml("boston", version=1)
print(type(boston))
