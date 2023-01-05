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

""""

"""

csv_foldar_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/dining_hall/"
csv_foldar_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/elevator/"
csv_foldar_3 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/sota/"
csv_foldar_4 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/staff_station/"
csv_foldar_5 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/stairs/"
# csv_foldar_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1227_downsampler0.2_y<4/dining_hall/"

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
data_list_1 = []
data_list_2 = []
data_list_3 = []

#読み込むファイルのリストを走査
for file in files_1_1:
    data_list_1.append(pd.read_csv(file))

df = pd.concat(data_list_1, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_dining_hall/dining_hall_1.csv",index=False)

for file in files_1_2:
    data_list_2.append(pd.read_csv(file))
df = pd.concat(data_list_2, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_dining_hall/dining_hall_2.csv",index=False)
for file in files_1_3:
    data_list_3.append(pd.read_csv(file))
df = pd.concat(data_list_3, axis=0, sort=True)
df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/gmm_csv_3/total_dining_hall/dining_hall_3.csv",index=False)

