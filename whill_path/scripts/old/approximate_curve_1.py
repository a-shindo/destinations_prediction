import os
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
import ntpath
from sklearn import linear_model
"""
pip install scikit-learn
"""
import joblib
from scipy import optimize

""""
https://algorithm.joho.info/programming/python/approximate-straight-line-curve/
"""


# 近似関数
def approx_func(param, x, y):
    y = param[0] * x ** 3 + param[1] * x ** 2 + param[2] * x + param[3]
    return y

def residual_func(param, x, y):
    residual = y - approx_func(param, x, y)
    return residual

# 決定係数R^2の計算
def calc_r2(x_data, y_data, y_predict):
    residuals = y_data - y_predict
    rss = np.sum(residuals**2)  # residual sum of squares = rss
    tss = np.sum((y_data - np.mean(y_data)) ** 2)  # total sum of squares = tss 
    r2 = 1 - (rss / tss)
    return r2

# パラメータの初期値
param1 = [0, 0, 0, 0]

csvPath='/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2/20221215_170110.csv'

csv_name = os.path.splitext(os.path.basename(csvPath))[0]

df = pd.read_csv(csvPath)
data_x = df[df.columns[0]]
data_y = df[df.columns[1]]


# 近似直線のパラメータを計算
lq = optimize.leastsq(residual_func, param1, args=(data_x, data_y))  # 係数
(a1, a2, a3, b) = (lq[0][0], lq[0][1], lq[0][2], lq[0][3])

# 近似直線
y_predict = approx_func(lq[0], data_x, data_y)

# 決定係数
r2 = calc_r2(data_x, data_y, y_predict)

# パラメータの表示
print("回帰係数a1:", a1)
print("回帰係数a2:", a2)
print("回帰係数a3:", a3)
print("切片:", b)
print("決定係数:", r2)

plt.plot(data_x, data_y)
plt.plot(data_x, y_predict)

plt.show()