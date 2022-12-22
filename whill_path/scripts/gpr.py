import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from glob import glob
from os.path import join
import random
import pandas as pd


"""

"""

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
print("test_2", test_2)
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

# ガウスカーネルを関数化
def kernel(x, x_prime, p, q, r):
    if x == x_prime:
        delta = 1
    else:
        delta = 0

    return p*np.exp(-1 * (x - x_prime)**2 / q) + ( r * delta)

# データの定義
xtrain = np.copy(training_data1["x"])
ytrain = np.copy(training_data1["y"])

xtest = np.copy(test_1["x"])


# 平均
mu = []
# 分散
var = []

# 各パラメータ値
Theta_1 = 1.0
Theta_2 = 0.4
Theta_3 = 0.1

# 以下, ガウス過程回帰の計算の基本アルゴリズム
train_length = len(xtrain)
# トレーニングデータ同士のカーネル行列の下地を準備
K = np.zeros((train_length, train_length))

for x in range(train_length):
    for x_prime in range(train_length):
        K[x, x_prime] = kernel(xtrain[x], xtrain[x_prime], Theta_1, Theta_2, Theta_3)

# 内積はドットで計算
yy = np.dot(np.linalg.inv(K), ytrain)

test_length = len(xtest)
for x_test in range(test_length):

    # テストデータとトレーニングデータ間のカーネル行列の下地を準備
    k = np.zeros((train_length,))
    for x in range(train_length):
        k[x] = kernel(xtrain[x], xtest[x_test], Theta_1, Theta_2, Theta_3)

    s = kernel(xtest[x_test], xtest[x_test], Theta_1, Theta_2, Theta_3)

    # 内積はドットで計算して, 平均値の配列に追加
    mu.append(np.dot(k, yy))
    # 先に『k * K^-1』の部分を(内積なのでドットで)計算
    kK_ = np.dot(k, np.linalg.inv(K))
    # 後半部分との内積をドットで計算して, 分散の配列に追加
    var.append(s - np.dot(kK_, k.T))




# fig = plt.figure(figsize=[28,14])
# ax0=fig.add_subplot(141) #(figsize=[21,14])
# ax0.set_xlim(1, 7)
# ax0.set_ylim(1, 13)
# # ax1.contour(XX_1, YY_1, pdf1, cmap='Blues',zorder=3)
# # # plt.colorbar() # カラーバー
# # ax1.contour(XX_2, YY_2, pdf2, cmap='Reds',zorder=4)
# # plt.colorbar() 
# ax0.scatter(training_data1["x"], training_data1["y"], s=4, c="lightblue",zorder=2)
# ax0.scatter(training_data2["x"], training_data2["y"], s=4, c="pink",zorder=1)
# ax0.set_xlabel('x', size=10)
# ax0.set_ylabel('y', size=10)

plt.figure(figsize=(6, 12))
plt.title('signal prediction by Gaussian process', fontsize=20)

# 元の信号
plt.plot(training_data1["x"], training_data1["y"], 'x', color='green', label='correct signal')
# 部分的なサンプル点
plt.plot(test_1["x"], test_1["y"], 'o', color='red', label='sample dots')



# 分散を標準偏差に変換
std = np.sqrt(var)

# ガウス過程で求めた平均値を信号化
plt.plot(xtest, mu, color='blue', label='mean by Gaussian process')
# ガウス過程で求めた標準偏差を範囲化 *範囲に関してはコード末を参照
plt.fill_between(xtest, mu + 2*std, mu - 2*std, alpha=.2, color='blue', label= 'standard deviation by Gaussian process')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
plt.show()
