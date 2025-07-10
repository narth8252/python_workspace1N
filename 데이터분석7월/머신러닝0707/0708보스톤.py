from sklearn.datasets import fetch_openml
boston = fetch_openml("boston", version=1)
print(type(boston))
print(boston.keys())
print(boston["DESCR"])

X = boston["data"]
y = boston["target"]

print(X.shape)
print(X[:10])
print(y[:10])
"""
<class 'sklearn.utils._bunch.Bunch'>
dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])
**Author**:
**Source**: Unknown - Date unknown
**Please cite**:

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.
Variables in order:
CRIM     per capita crime rate by town
ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS    proportion of non-retail business acres per town
CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX      nitric oxides concentration (parts per 10 million)
RM       average number of rooms per dwelling
AGE      proportion of owner-occupied units built prior to 1940
DIS      weighted distances to five Boston employment centres
RAD      index of accessibility to radial highways
TAX      full-value property-tax rate per $10,000
PTRATIO  pupil-teacher ratio by town
B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT    % lower status of the population
MEDV     Median value of owner-occupied homes in $1000's


Information about the dataset
CLASSTYPE: numeric
CLASSINDEX: last

Downloaded from openml.org.
(506, 13)
      CRIM    ZN  INDUS CHAS    NOX     RM    AGE     DIS RAD    TAX  PTRATIO       B  LSTAT
0  0.00632  18.0   2.31    0  0.538  6.575   65.2  4.0900   1  296.0     15.3  396.90   4.98
1  0.02731   0.0   7.07    0  0.469  6.421   78.9  4.9671   2  242.0     17.8  396.90   9.14
2  0.02729   0.0   7.07    0  0.469  7.185   61.1  4.9671   2  242.0     17.8  392.83   4.03
3  0.03237   0.0   2.18    0  0.458  6.998   45.8  6.0622   3  222.0     18.7  394.63   2.94
4  0.06905   0.0   2.18    0  0.458  7.147   54.2  6.0622   3  222.0     18.7  396.90   5.33
5  0.02985   0.0   2.18    0  0.458  6.430   58.7  6.0622   3  222.0     18.7  394.12   5.21
6  0.08829  12.5   7.87    0  0.524  6.012   66.6  5.5605   5  311.0     15.2  395.60  12.43
7  0.14455  12.5   7.87    0  0.524  6.172   96.1  5.9505   5  311.0     15.2  396.90  19.15
8  0.21124  12.5   7.87    0  0.524  5.631  100.0  6.0821   5  311.0     15.2  386.63  29.93
9  0.17004  12.5   7.87    0  0.524  6.004   85.9  6.5921   5  311.0     15.2  386.71  17.10
0    24.0
1    21.6
2    34.7
3    33.4
4    36.2
5    28.7
6    22.9
7    27.1
8    16.5
9    18.9
Name: MEDV, dtype: float64
"""

