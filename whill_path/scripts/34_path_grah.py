import os
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math

""""
https://coffee-blue-mountain.com/python-for-csv-modi1/
"""

resolution = 0.2
# resolution = 0.1

plt.figure(figsize=[10,32])



path = Path('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28/stairs')
# path = Path('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal_34_1_20221227.28/dining_hall')



for i, file in enumerate(path.glob('*.csv')):
    file_index= i

    df = pd.read_csv(file)

    last_x = df["x"][0]
    last_y = df["y"][0]

    c = ['x', 'y']
    downsampled_df = pd.DataFrame(columns=c)
    downsampled_x = [last_x]
    downsampled_y = [last_y]

    for index, row in df.iterrows():
        x = row["x"]
        y = row["y"]

        if math.hypot(x - last_x, y - last_y) >= resolution:
            downsampled_x.append(x)
            downsampled_y.append(y)
            last_x = x
            last_y = y

    downsampled_df["x"] = downsampled_x
    downsampled_df["y"] = downsampled_y



    downsampled_df.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts//goal_34_1_20221227.28_downsampler0.2/stairs/'+"downsampled_" + str(file_index)+'.csv', header=True, index=False)
    # downsampled_df.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts//goal_34_1_20221227.28_downsampler0.2/dining_hall/'+"downsampled_" + str(file_index)+'.csv', header=True, index=False)
    


    plt.scatter(downsampled_df["x"], downsampled_df["y"], s=8, c="violet")
    # c= orange, violet, yellowgreen, red, light blue
    # plt.plot(downsampled_df["x"], downsampled_df["y"])
    plt.xlim(-4, 6)
    plt.ylim(-16, 16)
    
plt.show()



 
