import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

"""
https://www.anarchive-beta.com/entry/2021/05/25/075812
"""

data1, data2 = pd.read_csv("./total1.csv"), pd.read_csv("./total2.csv")

mean_xy_1, mean_xy_2 = np.mean(data1, 0),np.mean(data2, 0)
cov_xy_1 = np.cov(data1, rowvar=False)
cov_xy_2 = np.cov(data2, rowvar=False)
# Sigma11, Sigma12, Sigma21, Sigma22 = cov_xy.reshape(-1)
X_1 = np.linspace(np.min(data1["x"])-1,np.max(data1["x"])+1)
Y_1 = np.linspace(np.min(data1["y"])-1,np.max(data1["y"])+1)
XX_1, YY_1 = np.meshgrid(X_1,Y_1)
z_1 = np.dstack((XX_1, YY_1))
pdf1 = multivariate_normal.pdf(z_1, mean_xy_1, cov_xy_1)
# print("pdf1", pdf1)

X_2 = np.linspace(np.min(data2["x"])-1,np.max(data2["x"])+1)
Y_2 = np.linspace(np.min(data2["y"])-1,np.max(data2["y"])+1)
XX_2, YY_2 = np.meshgrid(X_2,Y_2)
z_2 = np.dstack((XX_2, YY_2))
pdf2 = multivariate_normal.pdf(z_2, mean_xy_2, cov_xy_2)
# print("pdf2", pdf2)

# 次元数を設定:(固定)
D = 2

# クラスタ数を指定
K = 2

# K個の真の平均を指定
mu_truth_kd = np.array(
    [mean_xy_1, 
    mean_xy_2]
)

# K個の真の共分散行列を指定
sigma2_truth_kdd = np.array(
    [cov_xy_1, 
     cov_xy_2]
)

# 真の混合係数を指定
pi_truth_k = np.array([0.5, 0.5])

# 確認
print("mu_truth_kd",mu_truth_kd)
print("sigma2_truth_kdd", sigma2_truth_kdd)

# 作図用のx軸のxの値を作成
x_1_line = np.linspace(
    np.min(mu_truth_kd[:, 0] - 2 * np.sqrt(sigma2_truth_kdd[:, 0])), 
    np.max(mu_truth_kd[:, 0] + 2 * np.sqrt(sigma2_truth_kdd[:, 0])), 
    num=300
)

# 作図用のy軸のxの値を作成
x_2_line = np.linspace(
    np.min(mu_truth_kd[:, 1] - 3 * np.sqrt(sigma2_truth_kdd[:, 1])), 
    np.max(mu_truth_kd[:, 1] + 3 * np.sqrt(sigma2_truth_kdd[:, 1])), 
    num=300
)

# 作図用の格子状の点を作成
x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)

# 作図用のxの点を作成
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)

# 作図用に各次元の要素数を保存
x_dim = x_1_grid.shape
print(x_dim)

# 真の分布を計算
model_dens = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_dens = multivariate_normal.pdf(
        x=x_point_arr, mean=mu_truth_kd[k], cov=sigma2_truth_kdd[k]
    )
    
    # K個の分布を線形結合
    model_dens += pi_truth_k[k] * tmp_dens




plt.figure(figsize=[6,10])
plt.xlim(1, 7)
plt.ylim(1, 13)
plt.contour(x_1_grid, x_2_grid, model_dens.reshape(x_dim)) # 真の分布
plt.colorbar() # カラーバー
plt.scatter(data1["x"], data1["y"], s=4, c="lightblue")
plt.scatter(data2["x"], data2["y"], s=4, c="pink")
# plt.title('visualization', size=20)
plt.xlabel('x', size=20)
plt.ylabel('y', size=20)
plt.show()