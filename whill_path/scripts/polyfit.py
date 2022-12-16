import os
from glob import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math


path = Path('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal1_downsampler/')


for i, file in enumerate(path.glob('*.csv')):

    df = pd.read_csv(file)
    data_x = df[df.columns[0]]
    data_y = df[df.columns[1]]
  
    plt.plot(data_x, data_y)
plt.show()