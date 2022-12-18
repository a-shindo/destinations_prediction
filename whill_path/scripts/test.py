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

#csvファイルの中身を追加していくリストを用意
data_list_1 = []

#読み込むファイルのリストを走査
for file in training_csv_1:
    data_list_1.append(pd.read_csv(file))

#リストを全て行方向に結合
#axis=0:行方向に結合, sort
df1 = pd.concat(data_list_1, axis=0, sort=True)

df1.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_training_1.csv",index=False)

data_list_2 = []
for file in training_csv_2:
    data_list_2.append(pd.read_csv(file))
df2 = pd.concat(data_list_2, axis=0, sort=True)
df2.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_training_2.csv",index=False)


print(test_csv_1[1])
 
# df = pd.read_csv(test[1])
# for index, data in df.iterrows():
#     # print(index)
#     # print(data)
#     print('--------')

# print(test[1])




training_data1, training_data2 = pd.read_csv("./total_training_1.csv"), pd.read_csv("./total_training_2.csv")

mean_xy_1, mean_xy_2 = np.mean(training_data1, 0),np.mean(training_data2, 0)
cov_xy_1 = np.cov(training_data1, rowvar=False)
cov_xy_2 = np.cov(training_data2, rowvar=False)
print("mean_xy_1, mean_xy_2 ", mean_xy_1, mean_xy_2 )
print("cov_xy", cov_xy_1,cov_xy_2)
# Sigma11, Sigma12, Sigma21, Sigma22 = cov_xy.reshape(-1)
X_1 = np.linspace(np.min(training_data1["x"])-1,np.max(training_data1["x"])+1)
Y_1 = np.linspace(np.min(training_data1["y"])-1,np.max(training_data1["y"])+1)
XX_1, YY_1 = np.meshgrid(X_1,Y_1)
z_1 = np.dstack((XX_1, YY_1))
pdf1 = multivariate_normal.pdf(z_1, mean_xy_1, cov_xy_1)
print("pdf1", pdf1)

X_2 = np.linspace(np.min(training_data2["x"])-1,np.max(training_data2["x"])+1)
Y_2 = np.linspace(np.min(training_data2["y"])-1,np.max(training_data2["y"])+1)
XX_2, YY_2 = np.meshgrid(X_2,Y_2)
z_2 = np.dstack((XX_2, YY_2))
pdf2 = multivariate_normal.pdf(z_2, mean_xy_2, cov_xy_2)
print("pdf2", pdf2)

plt.figure(figsize=[6,10])
plt.xlim(1, 7)
plt.ylim(1, 13)
plt.contour(XX_1, YY_1, pdf1, cmap='Blues')
plt.colorbar() # カラーバー
plt.contour(XX_2, YY_2, pdf2, cmap='Reds')
plt.colorbar() # カラーバー
plt.scatter(training_data1["x"], training_data1["y"], s=4, c="lightblue")
plt.scatter(training_data2["x"], training_data2["y"], s=4, c="pink")
plt.xlabel('x', size=20)
plt.ylabel('y', size=20)
plt.show()

