import os
from glob import glob
from os.path import join
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import multivariate_normal

""""
https://coffee-blue-mountain.com/python-for-csv-modi1/
"""


plt.figure(figsize=[10,32])

path = Path('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28_downsampler0.2/sota')


for i, file in enumerate(path.glob('*.csv')):
    file_index= i
    

    df = pd.read_csv(file)
    print("df, len(df)", df, len(df))
    
    first_x = df["x"][0]
    first_y = df["y"][0]

    c = ['x', 'y']
    df_1 = pd.DataFrame(columns=c)
    df_1_x = [first_x]
    df_1_y = [first_y]

    df_2 = df.tail(1)
    print("df_2", df_2)
    # print("df_2[x], df_2[y]", df_2["x"], df_2["y"])
    # df_2_x = [first_x]
    # df_1_y = [first_y]

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


    df_1.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/csv_3/sota/df_1/'+"df_1_" + str(file_index)+'.csv', header=True, index=False)
    df_2.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/csv_3/sota/df_2/'+"df_2_" + str(file_index)+'.csv', header=True, index=False)
    df_3.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/csv_3/sota/df_3/'+"df_3_" + str(file_index)+'.csv', header=True, index=False)

    # df_1.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts//goal_34_1_20221227.28_downsampler0.2/dining_hall, stairs, elevator, sota, staff_station, /'+"df_1_" + str(file_index)+'.csv', header=True, index=False)
    

    plt.scatter(df_1["x"], df_1["y"], s=8, c="lightblue")
    # plt.scatter(df_2[0], df_1[1], s=8, c="orange")
    # plt.scatter(df_3["x"], df_1["y"], s=8, c="violet")
    # c= orange, violet, yellowgreen, red, light blue
    # plt.plot(df_1["x"], df_1["y"])
    plt.xlim(-4, 6)
    plt.ylim(-16, 16)


plt.show()



 
