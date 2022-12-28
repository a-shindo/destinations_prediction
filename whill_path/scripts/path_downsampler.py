import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import math

resolution = 0.1

csvPath='/home/ytpc2017d/catkin_ws/src/whill_path/scripts/name_34_1_20221227.28/nakamurasan/20221227_181547.csv'

csv_name = os.path.splitext(os.path.basename(csvPath))[0]

df = pd.read_csv(csvPath)

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
# downsampled_df.to_csv(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/old_csv/34_1_20221227'+"downsampled_" + (csv_name)+'.csv', index=False)

plt.figure(figsize=(10,32))
plt.plot(downsampled_df["x"], downsampled_df["y"])
plt.xlim(-4, 6)
plt.ylim(-16, 16)
plt.show()