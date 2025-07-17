import pandas as pd
import numpy as np
import optuna
from sklearn.datasets import fetch_california_housing #"회귀Regression문제" 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

