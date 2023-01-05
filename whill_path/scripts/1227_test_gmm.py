import os
from glob import glob
from os.path import join
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import multivariate_normal
import random


csv_foldar_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/dining_hall/"
csv_foldar_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/elevator/"
csv_foldar_3 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/sota/"
csv_foldar_4 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/staff_station/"
csv_foldar_5 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/stairs/"

# 指定パスのファイルのリストを取得
files_1 = glob(join(csv_foldar_1, "*.csv"))
files_2 = glob(join(csv_foldar_2, "*.csv"))
files_3 = glob(join(csv_foldar_3, "*.csv"))
files_4 = glob(join(csv_foldar_4, "*.csv"))
files_5 = glob(join(csv_foldar_5, "*.csv"))
# ファイルの総数を取得
num_files_1 = len(files_1)
num_files_2 = len(files_2)
num_files_3 = len(files_3)
num_files_4 = len(files_4)
num_files_5 = len(files_5)

# ファイルのリストを2:19に分ける
test_csv_1 = random.sample(files_1, int(num_files_1*(2/19)))
training_csv_1 = random.sample(files_1, num_files_1 - int(num_files_1*(2/19)))
test_csv_2 = random.sample(files_2, int(num_files_2*(2/19)))
training_csv_2 = random.sample(files_2, num_files_2 - int(num_files_2*(2/19)))
test_csv_3 = random.sample(files_3, int(num_files_3*(2/19)))
training_csv_3 = random.sample(files_3, num_files_3 - int(num_files_3*(3/19)))
test_csv_4 = random.sample(files_4, int(num_files_4*(2/21)))
training_csv_4 = random.sample(files_4, num_files_4 - int(num_files_4*(2/21)))
test_csv_5 = random.sample(files_5, int(num_files_5*(2/19)))
training_csv_5 = random.sample(files_5, num_files_5 - int(num_files_5*(2/19)))
print("test_csv_1",test_csv_1)
print("test_csv_2",test_csv_2)
print("test_csv_3",test_csv_3)
print("test_csv_4",test_csv_4)
print("test_csv_5",test_csv_5)

# dining_hall
for i, file in enumerate(training_csv_1):
    file_index= i
    
    df = pd.read_csv(file)
    
    first_x = df["x"][0]
    first_y = df["y"][0]

    c = ['x', 'y']
    df_1 = pd.DataFrame(columns=c)
    df_1_x = [first_x]
    df_1_y = [first_y]

    df_2 = df.tail(1)
    # print("df_2", df_2)

    df_3 = pd.DataFrame(columns=c)
    next_x = []
    next_y = []

    for index, row in df.iterrows():
        x = row["x"]
        y = row["y"]

        if y >=3:
            df_1_x.append(x)
            df_1_y.append(y)
            first_x = x
            first_y = y

        elif index+1 <len(df):
            next_x.append(x)
            next_y.append(y)

    df_1["x"] = df_1_x
    df_1["y"] = df_1_y

    df_3_x = next_x
    df_3_y = next_y
    df_3["x"] = df_3_x
    df_3["y"] = df_3_y

    df_1.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/dining_hall/df_1/'+"df_1_" + str(file_index)+'.csv', header=True, index=False)
    df_2.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/dining_hall/df_2/'+"df_2_" + str(file_index)+'.csv', header=True, index=False)
    df_3.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/dining_hall/df_3/'+"df_3_" + str(file_index)+'.csv', header=True, index=False)

csv_folder_1_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/dining_hall/df_1/"
csv_folder_1_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/dining_hall/df_2/"
csv_folder_1_3 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/dining_hall/df_3/"
files_1_1 = glob(join(csv_folder_1_1, "*.csv"))
files_1_2 = glob(join(csv_folder_1_2, "*.csv"))
files_1_3 = glob(join(csv_folder_1_3, "*.csv"))

#csvファイルの中身を追加していくリストを用意
data_list_1_1 = []
data_list_1_2 = []
data_list_1_3 = []

#読み込むファイルのリストを走査
for file in files_1_1:
    data_list_1_1.append(pd.read_csv(file))

df = pd.concat(data_list_1_1, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_dining_hall/dining_hall_1.csv",index=False)

for file in files_1_2:
    data_list_1_2.append(pd.read_csv(file))
df = pd.concat(data_list_1_2, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_dining_hall/dining_hall_2.csv",index=False)
for file in files_1_3:
    data_list_1_3.append(pd.read_csv(file))
df = pd.concat(data_list_1_3, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_dining_hall/dining_hall_3.csv",index=False)


# elevator
for i, file in enumerate(training_csv_2):
    file_index= i
    df = pd.read_csv(file)
    first_x = df["x"][0]
    first_y = df["y"][0]
    c = ['x', 'y']
    df_1 = pd.DataFrame(columns=c)
    df_1_x = [first_x]
    df_1_y = [first_y]
    df_2 = df.tail(1)
    df_3 = pd.DataFrame(columns=c)
    next_x = []
    next_y = []

    for index, row in df.iterrows():
        x = row["x"]
        y = row["y"]

        if y >=3:
            df_1_x.append(x)
            df_1_y.append(y)
            first_x = x
            first_y = y

        elif index+1 <len(df):
            next_x.append(x)
            next_y.append(y)

    df_1["x"] = df_1_x
    df_1["y"] = df_1_y
    df_3_x = next_x
    df_3_y = next_y
    df_3["x"] = df_3_x
    df_3["y"] = df_3_y
    df_1.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/elevator/df_1/'+"df_1_" + str(file_index)+'.csv', header=True, index=False)
    df_2.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/elevator/df_2/'+"df_2_" + str(file_index)+'.csv', header=True, index=False)
    df_3.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/elevator/df_3/'+"df_3_" + str(file_index)+'.csv', header=True, index=False)

csv_folder_2_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/elevator/df_1/"
csv_folder_2_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/elevator/df_2/"
csv_folder_2_3 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/elevator/df_3/"
files_2_1 = glob(join(csv_folder_2_1, "*.csv"))
files_2_2 = glob(join(csv_folder_2_2, "*.csv"))
files_2_3 = glob(join(csv_folder_2_3, "*.csv"))

#csvファイルの中身を追加していくリストを用意
data_list_2_1 = []
data_list_2_2 = []
data_list_2_3 = []

#読み込むファイルのリストを走査
for file in files_2_1:
    data_list_2_1.append(pd.read_csv(file))
df = pd.concat(data_list_2_1, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_elevator/elevator_1.csv",index=False)

for file in files_2_2:
    data_list_2_2.append(pd.read_csv(file))
df = pd.concat(data_list_2_2, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_elevator/elevator_2.csv",index=False)
for file in files_2_3:
    data_list_2_3.append(pd.read_csv(file))
df = pd.concat(data_list_2_3, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_elevator/elevator_3.csv",index=False)

# sota
for i, file in enumerate(training_csv_3):
    file_index= i
    df = pd.read_csv(file)
    first_x = df["x"][0]
    first_y = df["y"][0]
    c = ['x', 'y']
    df_1 = pd.DataFrame(columns=c)
    df_1_x = [first_x]
    df_1_y = [first_y]
    df_2 = df.tail(1)
    df_3 = pd.DataFrame(columns=c)
    next_x = []
    next_y = []

    for index, row in df.iterrows():
        x = row["x"]
        y = row["y"]

        if y >=3:
            df_1_x.append(x)
            df_1_y.append(y)
            first_x = x
            first_y = y
        elif index+1 <len(df):
            next_x.append(x)
            next_y.append(y)

    df_1["x"] = df_1_x
    df_1["y"] = df_1_y

    df_3_x = next_x
    df_3_y = next_y
    df_3["x"] = df_3_x
    df_3["y"] = df_3_y

    df_1.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/sota/df_1/'+"df_1_" + str(file_index)+'.csv', header=True, index=False)
    df_2.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/sota/df_2/'+"df_2_" + str(file_index)+'.csv', header=True, index=False)
    df_3.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/sota/df_3/'+"df_3_" + str(file_index)+'.csv', header=True, index=False)

csv_folder_3_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/sota/df_1/"
csv_folder_3_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/sota/df_2/"
csv_folder_3_3 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/sota/df_3/"
files_3_1 = glob(join(csv_folder_3_1, "*.csv"))
files_3_2 = glob(join(csv_folder_3_2, "*.csv"))
files_3_3 = glob(join(csv_folder_3_3, "*.csv"))

data_list_3_1 = []
data_list_3_2 = []
data_list_3_3 = []
for file in files_3_1:
    data_list_3_1.append(pd.read_csv(file))
df = pd.concat(data_list_3_1, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_sota/sota_1.csv",index=False)
for file in files_3_2:
    data_list_3_2.append(pd.read_csv(file))
df = pd.concat(data_list_3_2, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_sota/sota_2.csv",index=False)
for file in files_3_3:
    data_list_3_3.append(pd.read_csv(file))
df = pd.concat(data_list_3_3, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_sota/sota_3.csv",index=False)

# staff_station
for i, file in enumerate(training_csv_4):
    file_index= i
    df = pd.read_csv(file)
    first_x = df["x"][0]
    first_y = df["y"][0]
    c = ['x', 'y']
    df_1 = pd.DataFrame(columns=c)
    df_1_x = [first_x]
    df_1_y = [first_y]
    df_2 = df.tail(1)
    df_3 = pd.DataFrame(columns=c)
    next_x = []
    next_y = []

    for index, row in df.iterrows():
        x = row["x"]
        y = row["y"]

        if y >=3:
            df_1_x.append(x)
            df_1_y.append(y)
            first_x = x
            first_y = y
        elif index+1 <len(df):
            next_x.append(x)
            next_y.append(y)

    df_1["x"] = df_1_x
    df_1["y"] = df_1_y
    df_3_x = next_x
    df_3_y = next_y
    df_3["x"] = df_3_x
    df_3["y"] = df_3_y
    df_1.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/staff_station/df_1/'+"df_1_" + str(file_index)+'.csv', header=True, index=False)
    df_2.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/staff_station/df_2/'+"df_2_" + str(file_index)+'.csv', header=True, index=False)
    df_3.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/staff_station/df_3/'+"df_3_" + str(file_index)+'.csv', header=True, index=False)

csv_folder_4_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/staff_station/df_1/"
csv_folder_4_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/staff_station/df_2/"
csv_folder_4_3 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/staff_station/df_3/"
files_4_1 = glob(join(csv_folder_4_1, "*.csv"))
files_4_2 = glob(join(csv_folder_4_2, "*.csv"))
files_4_3 = glob(join(csv_folder_4_3, "*.csv"))

data_list_4_1 = []
data_list_4_2 = []
data_list_4_3 = []
for file in files_4_1:
    data_list_4_1.append(pd.read_csv(file))
df = pd.concat(data_list_4_1, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_staff_station/staff_station_1.csv",index=False)
for file in files_4_2:
    data_list_4_2.append(pd.read_csv(file))
df = pd.concat(data_list_4_2, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_staff_station/staff_station_2.csv",index=False)
for file in files_4_3:
    data_list_4_3.append(pd.read_csv(file))
df = pd.concat(data_list_4_3, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_staff_station/staff_station_3.csv",index=False)

# stairs
for i, file in enumerate(training_csv_5):
    file_index= i
    df = pd.read_csv(file)
    first_x = df["x"][0]
    first_y = df["y"][0]
    c = ['x', 'y']
    df_1 = pd.DataFrame(columns=c)
    df_1_x = [first_x]
    df_1_y = [first_y]
    df_2 = df.tail(1)
    df_3 = pd.DataFrame(columns=c)
    next_x = []
    next_y = []

    for index, row in df.iterrows():
        x = row["x"]
        y = row["y"]

        if y >=3:
            df_1_x.append(x)
            df_1_y.append(y)
            first_x = x
            first_y = y
        elif index+1 <len(df):
            next_x.append(x)
            next_y.append(y)

    df_1["x"] = df_1_x
    df_1["y"] = df_1_y
    df_3_x = next_x
    df_3_y = next_y
    df_3["x"] = df_3_x
    df_3["y"] = df_3_y
    df_1.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/stairs/df_1/'+"df_1_" + str(file_index)+'.csv', header=True, index=False)
    df_2.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/stairs/df_2/'+"df_2_" + str(file_index)+'.csv', header=True, index=False)
    df_3.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/stairs/df_3/'+"df_3_" + str(file_index)+'.csv', header=True, index=False)

csv_folder_5_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/stairs/df_1/"
csv_folder_5_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/stairs/df_2/"
csv_folder_5_3 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/stairs/df_3/"
files_5_1 = glob(join(csv_folder_5_1, "*.csv"))
files_5_2 = glob(join(csv_folder_5_2, "*.csv"))
files_5_3 = glob(join(csv_folder_5_3, "*.csv"))

data_list_5_1 = []
data_list_5_2 = []
data_list_5_3 = []
for file in files_5_1:
    data_list_5_1.append(pd.read_csv(file))
df = pd.concat(data_list_5_1, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_stairs/stairs_1.csv",index=False)
for file in files_5_2:
    data_list_5_2.append(pd.read_csv(file))
df = pd.concat(data_list_5_2, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_stairs/stairs_2.csv",index=False)
for file in files_5_3:
    data_list_5_3.append(pd.read_csv(file))
df = pd.concat(data_list_5_3, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_stairs/stairs_3.csv",index=False)







# # トレーニング用データからガウス分布作成
# training_data1, training_data2, training_data3, training_data4, training_data5 = \
#     pd.read_csv("./total_training_dining_hall.csv"), pd.read_csv("./total_training_elevator.csv"), pd.read_csv("./total_training_sota.csv"), pd.read_csv("./total_training_staff_station.csv"), pd.read_csv("./total_training_stairs.csv")

# mean_xy_1, mean_xy_2, mean_xy_3, mean_xy_4, mean_xy_5 \
#     = np.mean(training_data1, 0),np.mean(training_data2, 0),np.mean(training_data3, 0),np.mean(training_data4, 0),np.mean(training_data5, 0)
# cov_xy_1 = np.cov(training_data1, rowvar=False)
# cov_xy_2 = np.cov(training_data2, rowvar=False)
# cov_xy_3 = np.cov(training_data3, rowvar=False)
# cov_xy_4 = np.cov(training_data4, rowvar=False)
# cov_xy_5 = np.cov(training_data5, rowvar=False)



# X_1 = np.linspace(np.min(training_data1["x"])-1,np.max(training_data1["x"])+1)
# Y_1 = np.linspace(np.min(training_data1["y"])-1,np.max(training_data1["y"])+1)
# XX_1, YY_1 = np.meshgrid(X_1,Y_1)
# z_1 = np.dstack((XX_1, YY_1))
# pdf1 = multivariate_normal.pdf(z_1, mean_xy_1, cov_xy_1)
# # print("pdf1", pdf1, len(pdf1),len(pdf1[2]))
# # len(pdf1[1])



# X_2 = np.linspace(np.min(training_data2["x"])-1,np.max(training_data2["x"])+1)
# Y_2 = np.linspace(np.min(training_data2["y"])-1,np.max(training_data2["y"])+1)
# XX_2, YY_2 = np.meshgrid(X_2,Y_2)
# z_2 = np.dstack((XX_2, YY_2))
# pdf2 = multivariate_normal.pdf(z_2, mean_xy_2, cov_xy_2)

# X_3 = np.linspace(np.min(training_data3["x"])-1,np.max(training_data3["x"])+1)
# Y_3 = np.linspace(np.min(training_data3["y"])-1,np.max(training_data3["y"])+1)
# XX_3, YY_3 = np.meshgrid(X_3,Y_3)
# z_3 = np.dstack((XX_3, YY_3))
# pdf3 = multivariate_normal.pdf(z_3, mean_xy_3, cov_xy_3)

# X_4 = np.linspace(np.min(training_data4["x"])-1,np.max(training_data4["x"])+1)
# Y_4 = np.linspace(np.min(training_data4["y"])-1,np.max(training_data4["y"])+1)
# XX_4, YY_4 = np.meshgrid(X_4,Y_4)
# z_4 = np.dstack((XX_4, YY_4))
# pdf4 = multivariate_normal.pdf(z_4, mean_xy_4, cov_xy_4)

# X_5 = np.linspace(np.min(training_data5["x"])-1,np.max(training_data5["x"])+1)
# Y_5 = np.linspace(np.min(training_data5["y"])-1,np.max(training_data5["y"])+1)
# XX_5, YY_5 = np.meshgrid(X_5,Y_5)
# z_5 = np.dstack((XX_5, YY_5))
# pdf5 = multivariate_normal.pdf(z_5, mean_xy_5, cov_xy_5)
# # print("pdf2", pdf2)

# # plt.figure(figsize=[14,14])

# z_test_list = pd.read_csv(test_csv_1[0]).values.tolist()
# print("test_csv_2", test_csv_2)
# # print("z_test_list",z_test_list[1],len(pd.read_csv(test_csv_2[0]).index))

# num=0
# pdf1_test=[]
# pdf2_test=[]
# pdf3_test=[]
# pdf4_test=[]
# pdf5_test=[]
# nsa_pdf1_test=[]
# nsa_pdf2_test=[]
# nsa_pdf3_test=[]
# nsa_pdf4_test=[]
# nsa_pdf5_test=[]
# num1=1
# num2=1
# num3=1
# num4=1
# num5=1
# pi_pdf1_test=[]
# pi_pdf2_test=[]
# pi_pdf3_test=[]
# pi_pdf4_test=[]
# pi_pdf5_test=[]
# nsa_pi_pdf1_test=[]
# nsa_pi_pdf2_test=[]
# nsa_pi_pdf3_test=[]
# nsa_pi_pdf4_test=[]#csvファイルの中身を追加していくリストを用意
# nsa_pi_pdf5_test=[]
# while num < len(pd.read_csv(test_csv_1[0]).index):
#     pdf1_test_data = multivariate_normal.pdf(z_test_list[num], mean_xy_1, cov_xy_1)
#     pdf2_test_data = multivariate_normal.pdf(z_test_list[num], mean_xy_2, cov_xy_2)
#     pdf3_test_data = multivariate_normal.pdf(z_test_list[num], mean_xy_3, cov_xy_3)
#     pdf4_test_data = multivariate_normal.pdf(z_test_list[num], mean_xy_4, cov_xy_4)
#     pdf5_test_data = multivariate_normal.pdf(z_test_list[num], mean_xy_5, cov_xy_5)

#     pdf1_test.append([num, pdf1_test_data])
#     pdf2_test.append([num, pdf2_test_data])
#     pdf3_test.append([num, pdf3_test_data])
#     pdf4_test.append([num, pdf4_test_data])
#     pdf5_test.append([num, pdf5_test_data])


#     nsa_pdf1_test.append([num, pdf1_test_data/(pdf1_test_data+pdf2_test_data+pdf3_test_data+pdf4_test_data+pdf5_test_data)])# 正規化 
#     nsa_pdf2_test.append([num, pdf2_test_data/(pdf1_test_data+pdf2_test_data+pdf3_test_data+pdf4_test_data+pdf5_test_data)])
#     nsa_pdf3_test.append([num, pdf3_test_data/(pdf1_test_data+pdf2_test_data+pdf3_test_data+pdf4_test_data+pdf5_test_data)])
#     nsa_pdf4_test.append([num, pdf4_test_data/(pdf1_test_data+pdf2_test_data+pdf3_test_data+pdf4_test_data+pdf5_test_data)])
#     nsa_pdf5_test.append([num, pdf5_test_data/(pdf1_test_data+pdf2_test_data+pdf3_test_data+pdf4_test_data+pdf5_test_data)])

#     num1*= pdf1_test_data
#     num2*= pdf2_test_data
#     num3*= pdf3_test_data
#     num4*= pdf4_test_data
#     num5*= pdf5_test_data
#     pi_pdf1_test.append([num, num1])
#     nsa_pi_pdf1_test.append([num, num1/(num1+num2+num3+num4+num5)])
#     pi_pdf2_test.append([num, num2])
#     nsa_pi_pdf2_test.append([num, num2/(num1+num2+num3+num4+num5)])
#     pi_pdf3_test.append([num, num3])
#     nsa_pi_pdf3_test.append([num, num3/(num1+num2+num3+num4+num5)])
#     pi_pdf4_test.append([num, num4])
#     nsa_pi_pdf4_test.append([num, num4/(num1+num2+num3+num4+num5)])
#     pi_pdf5_test.append([num, num5])
#     nsa_pi_pdf5_test.append([num, num5/(num1+num2+num3+num4+num5)])
#     # color_list.append(num/(len(pd.read_csv(test_csv_1[0]).index)+1))
#     num+=1
# # print("pdf1_test",pdf1_test)
# # print("pi_pdf1_test",pi_pdf1_test)
# # print("nsa_pdf1_test",nsa_pdf1_test)
# # print("nsa_pi_pdf1_test",nsa_pi_pdf1_test)
# # print("[r[0] for r in z_test_list]", [r[0] for r in z_test_list])

# # for num in range(len(pd.read_csv(test_csv_1[0]))+1):
# #     colormap = num/len(pd.read_csv(test_csv_1[0]))
# fig = plt.figure(figsize=[28,14])
# ax0=fig.add_subplot(141) #(figsize=[21,14])
# ax0.set_xlim(-4, 6)
# ax0.set_ylim(-16, 16)
# # ax1.contour(XX_1, YY_1, pdf1, cmap='Blues',zorder=3)
# # # plt.colorbar() # カラーバー
# # ax1.contour(XX_2, YY_2, pdf2, cmap='Reds',zorder=4)
# # plt.colorbar() 
# ax0.scatter(training_data1["x"], training_data1["y"], s=2, c="orange")
# ax0.scatter(training_data2["x"], training_data2["y"], s=2, c="yellowgreen")
# ax0.scatter(training_data3["x"], training_data3["y"], s=2, c="red")
# ax0.scatter(training_data4["x"], training_data4["y"], s=2, c="lightblue")
# ax0.scatter(training_data5["x"], training_data5["y"], s=2, c="mediumpurple")
# ax0.set_xlabel('x', size=10)
# ax0.set_ylabel('y', size=10)


# ax1=fig.add_subplot(142) #(figsize=[21,14])
# ax1.set_xlim(-4, 6)
# ax1.set_ylim(-16, 16)
# ax1.contour(XX_1, YY_1, pdf1, cmap='Oranges')
# # plt.colorbar() # カラーバー
# ax1.contour(XX_2, YY_2, pdf2, cmap='YlGn')
# ax1.contour(XX_3, YY_3, pdf3, cmap='Reds')
# ax1.contour(XX_4, YY_4, pdf4, cmap='Blues')
# ax1.contour(XX_5, YY_5, pdf5, cmap='Purples')

# ax1.scatter(training_data1["x"], training_data1["y"], s=2, c="orange")
# ax1.scatter(training_data2["x"], training_data2["y"], s=2, c="yellowgreen")
# ax1.scatter(training_data3["x"], training_data3["y"], s=2, c="red")
# ax1.scatter(training_data4["x"], training_data4["y"], s=2, c="lightblue")
# ax1.scatter(training_data5["x"], training_data5["y"], s=2, c="mediumpurple")
# ax1.set_xlabel('x', size=10)
# ax1.set_ylabel('y', size=10)

# # cm=plt.get_cmap('Blues') 
# # cm_interval=[ i / (len(pd.read_csv(test_csv_1[0]).index)) for i in range(0, len(pd.read_csv(test_csv_1[0]).index)) ] 
# # print("cm_interval",cm_interval) 
# # cm=cm(cm_interval)

# def generate_cmap(colors):
#     """自分で定義したカラーマップを返す"""
#     values = range(len(colors))

#     vmax = np.ceil(np.max(values))
#     color_list = []
#     for v, c in zip(values, colors):
#         color_list.append( ( v/ vmax, c) )
#     return LinearSegmentedColormap.from_list('custom_cmap', color_list)

# colormap = generate_cmap(['blue', 'silver']) 
# traj_z_test_list_x = [r[0] for r in z_test_list]
# traj_z_test_list_y = [r[1] for r in z_test_list]       
# t = np.linspace(0,1,len(traj_z_test_list_x))
# cm = colormap(t)

# for j in range(len(traj_z_test_list_x)-1):
#     ax1.plot(traj_z_test_list_x[j:j+2], traj_z_test_list_y[j:j+2], color = cm[j], marker='o')
#     ax0.plot(traj_z_test_list_x[j:j+2], traj_z_test_list_y[j:j+2], color = cm[j], marker='.')
# # ax1.plot([r[0] for r in z_test_list],[r[1] for r in z_test_list], marker='.',  zorder=5)


# ax2=fig.add_subplot(247)
# traj_pdf1_test_x = [r[0] for r in pdf1_test]
# traj_pdf1_test_y = [r[1] for r in pdf1_test] 
# traj_pdf2_test_x = [r[0] for r in pdf2_test] 
# traj_pdf2_test_y = [r[1] for r in pdf2_test]
# traj_pdf3_test_x = [r[0] for r in pdf3_test] 
# traj_pdf3_test_y = [r[1] for r in pdf3_test]
# traj_pdf4_test_x = [r[0] for r in pdf4_test] 
# traj_pdf4_test_y = [r[1] for r in pdf4_test]
# traj_pdf5_test_x = [r[0] for r in pdf5_test] 
# traj_pdf5_test_y = [r[1] for r in pdf5_test]
# for j in range(len(traj_pdf1_test_x)-1):
#     ax2.scatter(traj_pdf1_test_x[j:j+2], traj_pdf1_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax2.scatter(traj_pdf2_test_x[j:j+2], traj_pdf2_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax2.scatter(traj_pdf3_test_x[j:j+2], traj_pdf3_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax2.scatter(traj_pdf4_test_x[j:j+2], traj_pdf4_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax2.scatter(traj_pdf5_test_x[j:j+2], traj_pdf5_test_y[j:j+2], color = cm[j], s=10,zorder=6)
# ax2.plot(traj_pdf1_test_x,[r[1] for r in pdf1_test],color='orange', linewidth = 4.0,zorder=1)
# ax2.plot([r[0] for r in pdf2_test],[r[1] for r in pdf2_test],color='yellowgreen',linewidth = 4.0,zorder=1)
# ax2.plot([r[0] for r in pdf3_test],[r[1] for r in pdf3_test],color='red',linewidth = 4.0,zorder=1)
# ax2.plot([r[0] for r in pdf4_test],[r[1] for r in pdf4_test],color='lightblue',linewidth = 4.0,zorder=1)
# ax2.plot([r[0] for r in pdf5_test],[r[1] for r in pdf5_test],color='mediumpurple',linewidth = 4.0,zorder=1)
# # 
# ax3=fig.add_subplot(243)
# ax3.set_yscale('log')
# traj_pi_pdf1_test_x = [r[0] for r in pi_pdf1_test]
# traj_pi_pdf1_test_y = [r[1] for r in pi_pdf1_test]
# traj_pi_pdf2_test_x = [r[0] for r in pi_pdf2_test] 
# traj_pi_pdf2_test_y = [r[1] for r in pi_pdf2_test]
# traj_pi_pdf3_test_x = [r[0] for r in pi_pdf3_test] 
# traj_pi_pdf3_test_y = [r[1] for r in pi_pdf3_test]
# traj_pi_pdf4_test_x = [r[0] for r in pi_pdf4_test] 
# traj_pi_pdf4_test_y = [r[1] for r in pi_pdf4_test]
# traj_pi_pdf5_test_x = [r[0] for r in pi_pdf5_test] 
# traj_pi_pdf5_test_y = [r[1] for r in pi_pdf5_test]
# for j in range(len(traj_pdf1_test_x)-1):
#     ax3.scatter(traj_pi_pdf1_test_x[j:j+2], traj_pi_pdf1_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax3.scatter(traj_pi_pdf2_test_x[j:j+2], traj_pi_pdf2_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax3.scatter(traj_pi_pdf3_test_x[j:j+2], traj_pi_pdf3_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax3.scatter(traj_pi_pdf4_test_x[j:j+2], traj_pi_pdf4_test_y[j:j+2], color = cm[j], s=10,zorder=6)
#     ax3.scatter(traj_pi_pdf5_test_x[j:j+2], traj_pi_pdf5_test_y[j:j+2], color = cm[j], s=10,zorder=6)
# ax3.plot([r[0] for r in pi_pdf1_test],[r[1] for r in pi_pdf1_test], color='orange', linewidth = 4.0, zorder=1)
# ax3.plot([r[0] for r in pi_pdf2_test],[r[1] for r in pi_pdf2_test], color='yellowgreen',linewidth = 4.0, zorder=1)
# ax3.plot([r[0] for r in pi_pdf3_test],[r[1] for r in pi_pdf3_test], color='red',linewidth = 4.0, zorder=1)
# ax3.plot([r[0] for r in pi_pdf4_test],[r[1] for r in pi_pdf4_test], color='lightblue',linewidth = 4.0, zorder=1)
# ax3.plot([r[0] for r in pi_pdf5_test],[r[1] for r in pi_pdf5_test], color='mediumpurple',linewidth = 4.0, zorder=1)


# ax4=fig.add_subplot(248)
# traj_nsa_pdf1_test_x = [r[0] for r in nsa_pdf1_test]
# traj_nsa_pdf1_test_y = [r[1] for r in nsa_pdf1_test] 
# traj_nsa_pdf2_test_x = [r[0] for r in nsa_pdf2_test] 
# traj_nsa_pdf2_test_y = [r[1] for r in nsa_pdf2_test]
# traj_nsa_pdf3_test_x = [r[0] for r in nsa_pdf3_test] 
# traj_nsa_pdf3_test_y = [r[1] for r in nsa_pdf3_test]
# traj_nsa_pdf4_test_x = [r[0] for r in nsa_pdf4_test] 
# traj_nsa_pdf4_test_y = [r[1] for r in nsa_pdf4_test]
# traj_nsa_pdf5_test_x = [r[0] for r in nsa_pdf5_test] 
# traj_nsa_pdf5_test_y = [r[1] for r in nsa_pdf5_test]
# for j in range(len(traj_pdf1_test_x)-1):
#     ax4.scatter(traj_nsa_pdf1_test_x[j:j+2], traj_nsa_pdf1_test_y[j:j+2], color = cm[j], s=10,zorder=3)
#     ax4.scatter(traj_nsa_pdf2_test_x[j:j+2], traj_nsa_pdf2_test_y[j:j+2], color = cm[j], s=10,zorder=4)
#     ax4.scatter(traj_nsa_pdf3_test_x[j:j+2], traj_nsa_pdf3_test_y[j:j+2], color = cm[j], s=10,zorder=4)
#     ax4.scatter(traj_nsa_pdf4_test_x[j:j+2], traj_nsa_pdf4_test_y[j:j+2], color = cm[j], s=10,zorder=4)
#     ax4.scatter(traj_nsa_pdf5_test_x[j:j+2], traj_nsa_pdf5_test_y[j:j+2], color = cm[j], s=10,zorder=4)
# ax4.plot(traj_nsa_pdf1_test_x,traj_nsa_pdf1_test_y, color='orange', linewidth = 4.0, zorder=1)
# ax4.plot(traj_nsa_pdf2_test_x,traj_nsa_pdf2_test_y, color='yellowgreen',linewidth = 4.0, zorder=1)
# ax4.plot(traj_nsa_pdf3_test_x,traj_nsa_pdf3_test_y, color='red',linewidth = 4.0, zorder=1)
# ax4.plot(traj_nsa_pdf4_test_x,traj_nsa_pdf4_test_y, color='lightblue',linewidth = 4.0, zorder=1)
# ax4.plot(traj_nsa_pdf5_test_x,traj_nsa_pdf5_test_y, color='mediumpurple',linewidth = 4.0, zorder=1)

# ax5=fig.add_subplot(244)
# traj_nsa_pi_pdf1_test_x = [r[0] for r in nsa_pi_pdf1_test]
# traj_nsa_pi_pdf1_test_y = [r[1] for r in nsa_pi_pdf1_test] 
# traj_nsa_pi_pdf2_test_x = [r[0] for r in nsa_pi_pdf2_test] 
# traj_nsa_pi_pdf2_test_y = [r[1] for r in nsa_pi_pdf2_test]
# traj_nsa_pi_pdf3_test_x = [r[0] for r in nsa_pi_pdf3_test] 
# traj_nsa_pi_pdf3_test_y = [r[1] for r in nsa_pi_pdf3_test]
# traj_nsa_pi_pdf4_test_x = [r[0] for r in nsa_pi_pdf4_test] 
# traj_nsa_pi_pdf4_test_y = [r[1] for r in nsa_pi_pdf4_test]
# traj_nsa_pi_pdf5_test_x = [r[0] for r in nsa_pi_pdf5_test] 
# traj_nsa_pi_pdf5_test_y = [r[1] for r in nsa_pi_pdf5_test]
# for j in range(len(traj_pdf1_test_x)-1):
#     ax5.scatter(traj_nsa_pi_pdf1_test_x[j:j+2], traj_nsa_pi_pdf1_test_y[j:j+2], color = cm[j], s=10,zorder=3)
#     ax5.scatter(traj_nsa_pi_pdf2_test_x[j:j+2], traj_nsa_pi_pdf2_test_y[j:j+2], color = cm[j], s=10,zorder=4)
#     ax5.scatter(traj_nsa_pi_pdf3_test_x[j:j+2], traj_nsa_pi_pdf3_test_y[j:j+2], color = cm[j], s=10,zorder=4)
#     ax5.scatter(traj_nsa_pi_pdf4_test_x[j:j+2], traj_nsa_pi_pdf4_test_y[j:j+2], color = cm[j], s=10,zorder=4)
#     ax5.scatter(traj_nsa_pi_pdf5_test_x[j:j+2], traj_nsa_pi_pdf5_test_y[j:j+2], color = cm[j], s=10,zorder=4)
# ax5.plot(traj_nsa_pi_pdf1_test_x,traj_nsa_pi_pdf1_test_y, color='orange', linewidth = 4.0, zorder=1)
# ax5.plot(traj_nsa_pi_pdf2_test_x,traj_nsa_pi_pdf2_test_y, color='yellowgreen',linewidth = 4.0, zorder=1)
# ax5.plot(traj_nsa_pi_pdf3_test_x,traj_nsa_pi_pdf3_test_y, color='red',linewidth = 4.0, zorder=1)
# ax5.plot(traj_nsa_pi_pdf4_test_x,traj_nsa_pi_pdf4_test_y, color='lightblue',linewidth = 4.0, zorder=1)
# ax5.plot(traj_nsa_pi_pdf5_test_x,traj_nsa_pi_pdf5_test_y, color='mediumpurple',linewidth = 4.0, zorder=1)



# plt.show()

