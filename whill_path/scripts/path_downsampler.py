import pandas as pd
import matplotlib.pyplot as plt
import math

resolution = 0.1

csv_name = "/goal1/20221215_170110.csv"

df = pd.read_csv(csv_name)

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
downsampled_df.to_csv("downsampled_" + csv_name, index=False)

plt.plot(downsampled_df["x"], downsampled_df["y"])
plt.show()