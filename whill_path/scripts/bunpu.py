import os
from glob import glob
from os.path import join
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import multivariate_normal

csv_foldar= "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/staff_station/"
files_1 = glob(join(csv_foldar, "*.csv"))

data_list = []
for file in files_1:
    data_list.append(pd.read_csv(file))

#リストを全て行方向に結合
#axis=0:行方向に結合, sort
df = pd.concat(data_list, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_staff_station.csv",index=False)
# df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_staff_station.csv",index=False)

# トレーニング用データからガウス分布作成
data = pd.read_csv("./total_staff_station.csv")

mean_xy = np.mean(data, 0)
cov_xy = np.cov(data, rowvar=False)

X_1 = np.linspace(np.min(data["x"])-1,np.max(data["x"])+1)
Y_1 = np.linspace(np.min(data["y"])-1,np.max(data["y"])+1)
XX_1, YY_1 = np.meshgrid(X_1,Y_1)
z_1 = np.dstack((XX_1, YY_1))
pdf1 = multivariate_normal.pdf(z_1, mean_xy, cov_xy)

plt.figure(figsize=[10,32])
plt.scatter(data["x"], data["y"], s=8, c="lightblue")
# c= orange, violet, yellowgreen, red, light blue
# plt.plot(downsampled_df["x"], downsampled_df["y"])
plt.xlim(-4, 6)
plt.ylim(-16, 16)
# plt.contour(XX_1, YY_1, pdf1, cmap='Oranges')
# plt.contour(XX_1, YY_1, pdf1, cmap='YlGn')
# plt.contour(XX_1, YY_1, pdf1, cmap='Reds')
plt.contour(XX_1, YY_1, pdf1, cmap='Blues')
# plt.contour(XX_1, YY_1, pdf1, cmap='Purples')
plt.colorbar() # カラーバー
plt.show()
