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
# ファイルのリストを3:20に分ける
test_csv_1 = random.sample(files_1, int(num_files_1*(3/23)))
training_csv_1 = random.sample(files_1, num_files_1 - int(num_files_1*(3/23)))
test_csv_2 = random.sample(files_2, int(num_files_2*(3/23)))
training_csv_2 = random.sample(files_2, num_files_2 - int(num_files_2*(3/23)))
print("test_csv_1,type(test_csv_1)", test_csv_1,type(test_csv_1))
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

X_1 = np.linspace(np.min(training_data1["x"])-1,np.max(training_data1["x"])+1)
Y_1 = np.linspace(np.min(training_data1["y"])-1,np.max(training_data1["y"])+1)
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

# plt.figure(figsize=[14,14])

z_test_list = pd.read_csv(test_csv_1[0]).values.tolist()
print("z_test_list",z_test_list[0],len(pd.read_csv(test_csv_1[0]).index))

num=0
pdf1_test=[]
pdf2_test=[]
nsa_pdf1_test=[]
nsa_pdf2_test=[]
num1=1
num2=1
pi_pdf1_test=[]
pi_pdf2_test=[]
nsa_pi_pdf1_test=[]
nsa_pi_pdf2_test=[]
while num < len(pd.read_csv(test_csv_1[0]).index):
    pdf1_test_data = multivariate_normal.pdf(z_test_list[num], mean_xy_1, cov_xy_1)
    pdf2_test_data = multivariate_normal.pdf(z_test_list[num], mean_xy_2, cov_xy_2)
    pdf1_test.append([num, pdf1_test_data])
    pdf2_test.append([num, pdf2_test_data])
    nsa_pdf1_test.append([num, pdf1_test_data/(pdf1_test_data+pdf2_test_data)])# 正規化 
    nsa_pdf2_test.append([num, pdf2_test_data/(pdf1_test_data+pdf2_test_data)])
    num1*= pdf1_test_data
    pi_pdf1_test.append([num, num1])
    nsa_pi_pdf1_test.append([num, num1/(num1+num2)])
    num2*= pdf2_test_data
    pi_pdf2_test.append([num, num2])
    nsa_pi_pdf2_test.append([num, num2/(num1+num2)])
    # color_list.append(num/(len(pd.read_csv(test_csv_1[0]).index)+1))
    num+=1
print("pdf1_test",pdf1_test)
print("pi_pdf1_test",pi_pdf1_test)
print("nsa_pdf1_test",nsa_pdf1_test)
print("nsa_pi_pdf1_test",nsa_pi_pdf1_test)
print("[r[0] for r in z_test_list]", [r[0] for r in z_test_list])

# for num in range(len(pd.read_csv(test_csv_1[0]))+1):
#     colormap = num/len(pd.read_csv(test_csv_1[0]))
fig = plt.figure(figsize=[28,14])
ax0=fig.add_subplot(141) #(figsize=[21,14])
ax0.set_xlim(1, 7)
ax0.set_ylim(1, 13)
# ax1.contour(XX_1, YY_1, pdf1, cmap='Blues',zorder=3)
# # plt.colorbar() # カラーバー
# ax1.contour(XX_2, YY_2, pdf2, cmap='Reds',zorder=4)
# plt.colorbar() 
ax0.scatter(training_data1["x"], training_data1["y"], s=4, c="lightblue",zorder=2)
ax0.scatter(training_data2["x"], training_data2["y"], s=4, c="pink",zorder=1)
ax0.set_xlabel('x', size=10)
ax0.set_ylabel('y', size=10)


ax1=fig.add_subplot(142) #(figsize=[21,14])
ax1.set_xlim(1, 7)
ax1.set_ylim(1, 13)
ax1.contour(XX_1, YY_1, pdf1, cmap='Blues',zorder=3)
# plt.colorbar() # カラーバー
ax1.contour(XX_2, YY_2, pdf2, cmap='Reds',zorder=4)
# plt.colorbar() 
ax1.scatter(training_data1["x"], training_data1["y"], s=4, c="lightblue",zorder=2)
ax1.scatter(training_data2["x"], training_data2["y"], s=4, c="pink",zorder=1)
ax1.set_xlabel('x', size=10)
ax1.set_ylabel('y', size=10)

# cm=plt.get_cmap('Blues') 
# cm_interval=[ i / (len(pd.read_csv(test_csv_1[0]).index)) for i in range(0, len(pd.read_csv(test_csv_1[0]).index)) ] 
# print("cm_interval",cm_interval) 
# cm=cm(cm_interval)

def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

colormap = generate_cmap(['green', 'yellow']) 
traj_z_test_list_x = [r[0] for r in z_test_list]
traj_z_test_list_y = [r[1] for r in z_test_list]       
t = np.linspace(0,1,len(traj_z_test_list_x))
cm = colormap(t)

for j in range(len(traj_z_test_list_x)-1):
    ax1.plot(traj_z_test_list_x[j:j+2], traj_z_test_list_y[j:j+2], color = cm[j], marker='o')
    ax0.plot(traj_z_test_list_x[j:j+2], traj_z_test_list_y[j:j+2], color = cm[j], marker='o')
# ax1.plot([r[0] for r in z_test_list],[r[1] for r in z_test_list], marker='.',  zorder=5)


ax2=fig.add_subplot(247)
traj_pdf1_test_x = [r[0] for r in pdf1_test]
traj_pdf1_test_y = [r[1] for r in pdf1_test] 
traj_pdf2_test_x = [r[0] for r in pdf2_test] 
traj_pdf2_test_y = [r[1] for r in pdf2_test]
for j in range(len(traj_pdf1_test_x)-1):
    ax2.scatter(traj_pdf1_test_x[j:j+2], traj_pdf1_test_y[j:j+2], color = cm[j], s=40,zorder=3)
    ax2.scatter(traj_pdf2_test_x[j:j+2], traj_pdf2_test_y[j:j+2], color = cm[j], s=40,zorder=4)
ax2.plot(traj_pdf1_test_x,[r[1] for r in pdf1_test],color='blue', linewidth = 2.0, zorder=1)
ax2.plot([r[0] for r in pdf2_test],[r[1] for r in pdf2_test],color='red',linewidth = 2.0, zorder=1)
# 
ax3=fig.add_subplot(243)
ax3.set_yscale('log')
traj_pi_pdf1_test_x = [r[0] for r in pi_pdf1_test]
traj_pi_pdf1_test_y = [r[1] for r in pi_pdf1_test]
traj_pi_pdf2_test_x = [r[0] for r in pi_pdf2_test] 
traj_pi_pdf2_test_y = [r[1] for r in pi_pdf2_test]
for j in range(len(traj_pdf1_test_x)-1):
    ax3.scatter(traj_pi_pdf1_test_x[j:j+2], traj_pi_pdf1_test_y[j:j+2], color = cm[j], s=40,zorder=3)
    ax3.scatter(traj_pi_pdf2_test_x[j:j+2], traj_pi_pdf2_test_y[j:j+2], color = cm[j], s=40,zorder=4)
ax3.plot([r[0] for r in pi_pdf1_test],[r[1] for r in pi_pdf1_test], color='blue', linewidth = 2.0, zorder=1)
ax3.plot([r[0] for r in pi_pdf2_test],[r[1] for r in pi_pdf2_test], color='red',linewidth = 2.0, zorder=1)

ax4=fig.add_subplot(248)
traj_nsa_pdf1_test_x = [r[0] for r in nsa_pdf1_test]
traj_nsa_pdf1_test_y = [r[1] for r in nsa_pdf1_test] 
traj_nsa_pdf2_test_x = [r[0] for r in nsa_pdf2_test] 
traj_nsa_pdf2_test_y = [r[1] for r in nsa_pdf2_test]
for j in range(len(traj_pdf1_test_x)-1):
    ax4.scatter(traj_nsa_pdf1_test_x[j:j+2], traj_nsa_pdf1_test_y[j:j+2], color = cm[j], s=40,zorder=3)
    ax4.scatter(traj_nsa_pdf2_test_x[j:j+2], traj_nsa_pdf2_test_y[j:j+2], color = cm[j], s=40,zorder=4)
ax4.plot(traj_nsa_pdf1_test_x,traj_nsa_pdf1_test_y, color='blue', linewidth = 2.0, zorder=1)
ax4.plot(traj_nsa_pdf2_test_x,traj_nsa_pdf2_test_y, color='red',linewidth = 2.0, zorder=1)

ax5=fig.add_subplot(244)
traj_nsa_pi_pdf1_test_x = [r[0] for r in nsa_pi_pdf1_test]
traj_nsa_pi_pdf1_test_y = [r[1] for r in nsa_pi_pdf1_test] 
traj_nsa_pi_pdf2_test_x = [r[0] for r in nsa_pi_pdf2_test] 
traj_nsa_pi_pdf2_test_y = [r[1] for r in nsa_pi_pdf2_test]
for j in range(len(traj_pdf1_test_x)-1):
    ax5.scatter(traj_nsa_pi_pdf1_test_x[j:j+2], traj_nsa_pi_pdf1_test_y[j:j+2], color = cm[j], s=40,zorder=3)
    ax5.scatter(traj_nsa_pi_pdf2_test_x[j:j+2], traj_nsa_pi_pdf2_test_y[j:j+2], color = cm[j], s=40,zorder=4)
ax5.plot(traj_nsa_pi_pdf1_test_x,traj_nsa_pi_pdf1_test_y, color='blue', linewidth = 2.0, zorder=1)
ax5.plot(traj_nsa_pi_pdf2_test_x,traj_nsa_pi_pdf2_test_y, color='red',linewidth = 2.0, zorder=1)



plt.show()












