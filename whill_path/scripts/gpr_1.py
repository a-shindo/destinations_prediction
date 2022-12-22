import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from glob import glob
from os.path import join
import random
import pandas as pd


np.random.seed(0)

def gpr(X_test, X_train, y_train, kernel):
    N = X_train.shape[0]
    K = np.empty([N, N])
    for n in range(N):
        for m in range(N):
            K[n, m] = kernel(X_train[n], X_train[m])
    K_inv = np.linalg.inv(K)
    yy = K_inv @ y_train
    M = X_test.shape[0]
    mu = np.empty(M)
    var = np.empty(M)
    for m in range(M):
        k = np.empty(N)
        for n in range(N):
            k[n] = kernel(X_train[n], X_test[m])
        s = kernel(X_test[m], X_test[m])
        mu[m] = np.dot(k, yy)
        var[m] = s - k @ K_inv @ np.transpose(k)
    return mu, var



csv_foldar_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal1_downsampler_0.2/"
csv_foldar_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2_downsampler_0.2/"

# 指定パスのファイルのリストを取得
files_1 = glob(join(csv_foldar_1, "*.csv"))
files_2 = glob(join(csv_foldar_2, "*.csv"))
# ファイルの総数を取得
num_files_1 = len(files_1)
num_files_2 = len(files_2)
# ファイルのリストを1:22に分ける
test_csv_1 = random.sample(files_1, int(num_files_1*(1/23)))
training_csv_1 = random.sample(files_1, num_files_1 - int(num_files_1*(1/23)))
test_csv_2 = random.sample(files_2, int(num_files_2*(1/23)))
training_csv_2 = random.sample(files_2, num_files_2 - int(num_files_2*(1/23)))
test_1=pd.read_csv(test_csv_1[0])
test_2=pd.read_csv(test_csv_2[0])
# print("test_2", test_2)
#csvファイルの中身を追加していくリストを用意
data_list_1 = []

#読み込むファイルのリストを走査
for file in training_csv_1:
    data_list_1.append(pd.read_csv(file))

#リストを全て行方向に結合
#axis=0:行方向に結合, sort
df_training_1 = pd.concat(data_list_1, axis=0, sort=True)
df_training_1.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_training_1.csv",index=False)

data_list_2 = []
for file in training_csv_2:
    data_list_2.append(pd.read_csv(file))
df_training_2 = pd.concat(data_list_2, axis=0, sort=True)
df_training_2.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_training_2.csv",index=False)

# 
training_data1, training_data2 = pd.read_csv("./total_training_1.csv"), pd.read_csv("./total_training_2.csv")



N = 1000
lb = 1.5
ub = 6
sigma = 0.5

def f(x):
    return np.sin(x)

# X_train = np.random.uniform(lb, ub, N)
X_train = np.copy(training_data1["x"])
# y_train = f(X_train) + np.random.normal(scale=sigma, size=N)
y_train = np.copy(training_data1["y"])
# plt.plot(X_train, y_train, "o")

def gauss_kernel(x1, x2, theta1=1.0, theta2=1.0):
    return theta1 * np.exp(-np.linalg.norm(x1 - x2) / theta2)


M = 50

X_test = np.linspace(lb, ub)
mu, var = gpr(X_test, X_train, y_train, gauss_kernel)
# # plt.plot(X_train, y_train, "o")
plt.figure(figsize=[5,11])
plt.plot(X_test, mu)
# plt.fill_between(X_test, mu - 2.0 * var, mu + 2.0 * var, color="navajowhite")

def gpr_with_noise(X_test, X_train, y_train, kernel, theta3=1.0):
    N = X_train.shape[0]
    K = np.empty([N, N])
    for n in range(N):
        K[n, n] = kernel(X_train[n], X_train[n]) + theta3
        for m in range(n+1, N):
            k = kernel(X_train[n], X_train[m])
            K[n, m] = k
            K[m, n] = k
    K_inv = np.linalg.inv(K)
    yy = K_inv @ y_train
    M = X_test.shape[0]
    mu = np.empty(M)
    var = np.empty(M)
    for m in range(M):
        k = np.empty(N)
        for n in range(N):
            k[n] = kernel(X_train[n], X_test[m])
        s = kernel(X_test[m], X_test[m]) + theta3
        mu[m] = np.dot(k, yy)
        var[m] = s - k @ K_inv @ np.transpose(k)
    return mu, var
mu, var = gpr_with_noise(X_test, X_train, y_train, gauss_kernel)
plt.plot(X_train, y_train, ".")
plt.plot(X_test, mu)
# plt.fill_between(X_test, mu - 2.0 * var, mu + 2.0 * var, color="navajowhite")
plt.xlim(1, 6)
plt.ylim(2, 13)
plt.show()
