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
import joblib
from scipy import optimize

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

path = Path('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal1_downsampler/')


for i, file in enumerate(path.glob('*.csv')):

    df = pd.read_csv(file)
    data_x = df[df.columns[0]]
    data_y = df[df.columns[1]]
    

    # 近似直線のパラメータを計算
    lq = optimize.leastsq(residual_func, param1, args=(times, Its))  # 係数
    (a1, a2, a3, b) = (lq[0][0], lq[0][1], lq[0][2], lq[0][3])

    plt.plot(data_x, data_y)
plt.show()