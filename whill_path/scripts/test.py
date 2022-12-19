import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.colors
# import matplotlib.cm as cm
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
# ファイルのリストを3:20に分ける
test_csv_1 = random.sample(files_1, int(num_files_1*(3/23)))
training_csv_1 = random.sample(files_1, num_files_1 - int(num_files_1*(3/23)))
test_csv_2 = random.sample(files_2, int(num_files_2*(3/23)))
training_csv_2 = random.sample(files_2, num_files_2 - int(num_files_2*(3/23)))
print(test_csv_1,type(test_csv_1))
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

# トレーニング用データからガウス分布作成
training_data1, training_data2 = pd.read_csv("./total_training_1.csv"), pd.read_csv("./total_training_2.csv")

mean_xy_1, mean_xy_2 = np.mean(training_data1, 0),np.mean(training_data2, 0)
cov_xy_1 = np.cov(training_data1, rowvar=False)
cov_xy_2 = np.cov(training_data2, rowvar=False)

X_1 = np.linspace(np.min(training_data1["x"])-1,np.max(training_data1["x"])+1,60)
Y_1 = np.linspace(np.min(training_data1["y"])-1,np.max(training_data1["y"])+1,60)
XX_1, YY_1 = np.meshgrid(X_1,Y_1)
z_1 = np.dstack((XX_1, YY_1))
pdf1 = multivariate_normal.pdf(z_1, mean_xy_1, cov_xy_1)
# print("pdf1", pdf1, len(pdf1),len(pdf1[2]))
# len(pdf1[1])



X_2 = np.linspace(np.min(training_data2["x"])-1,np.max(training_data2["x"])+1)
Y_2 = np.linspace(np.min(training_data2["y"])-1,np.max(training_data2["y"])+1)
XX_2, YY_2 = np.meshgrid(X_2,Y_2)
z_2 = np.dstack((XX_2, YY_2))
pdf2 = multivariate_normal.pdf(z_2, mean_xy_2, cov_xy_2)
# print("pdf2", pdf2)

plt.figure(figsize=[14,14])

test_data = pd.read_csv(test_csv_2[0])

x = test_data["x"].tolist()
y = test_data["y"].tolist()
num=0
num1=1
num2=1
while True:
    plt.subplot(121)
    if num == 0:
        plt.xlim(1, 7)
        plt.ylim(1, 13)
        plt.contour(XX_1, YY_1, pdf1, cmap='Blues',zorder=3)
        # plt.colorbar() # カラーバー
        plt.contour(XX_2, YY_2, pdf2, cmap='Reds',zorder=4)
        # plt.colorbar() # カラーバー
        plt.scatter(training_data1["x"], training_data1["y"], s=4, c="lightblue",zorder=2)
        plt.scatter(training_data2["x"], training_data2["y"], s=4, c="pink",zorder=1)
        plt.xlabel('x', size=20)
        plt.ylabel('y', size=20)
    plt.scatter(x[num],y[num], s=6,color='black',zorder=5)
    z_test=[x[num],y[num]]
    pdf1_test = multivariate_normal.pdf(z_test, mean_xy_1, cov_xy_1)
    pdf2_test = multivariate_normal.pdf(z_test, mean_xy_2, cov_xy_2)
    plt.subplot(224)
    plt.scatter(num,pdf1_test,color='blue')
    plt.scatter(num,pdf2_test,color='red')
    num1*= pdf1_test
    num2*= pdf2_test
    plt.subplot(222)
    plt.yscale('log')
    plt.scatter(num,num1,color='blue')
    plt.scatter(num,num2,color='red')
    num = num + 1
    if num>=len(test_data.index):
        break
    # plt.pause(0.01)
    # plt.show()


plt.show()












