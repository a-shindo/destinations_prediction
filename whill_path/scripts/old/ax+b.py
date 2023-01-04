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

csv_foldar_1 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal1/"
csv_foldar_2 = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2/"

# 指定パスのファイルのリストを取得
files_1 = glob(join(csv_foldar_1, "*.csv"))
files_2 = glob(join(csv_foldar_2, "*.csv"))
# ファイルの総数を取得
num_files_1 = len(files_1)
num_files_2 = len(files_2)
# ファイルのリストを1:22に分ける
test_csv_1 = random.sample(files_1, int(num_files_1*(1/23)))
training_csv_1 = random.sample(files_1, num_files_1 - int(num_files_1*(1/23)))
test_csv_2 = random.sample(files_2, int(num_files_2*(1/23)))
training_csv_2 = random.sample(files_2, num_files_2 - int(num_files_2*(1/23)))

# 目的地の位置（到達点の平均）
num1=0
num2=0
last_training_list_1=[]
while num1 < len(training_csv_1):
    def tail_pd_1(fn1, n):
        df = pd.read_csv(fn1)
        return df.tail(n).values.tolist()
    last_training_list_1.append(tail_pd_1(training_csv_1[num1], 1))
    num1+=1
last_training_list_2=[]
while num2 < len(training_csv_2):
    def tail_pd_2(fn2, n):
        df = pd.read_csv(fn2)
        return df.tail(n).values.tolist()
    last_training_list_2.append(tail_pd_2(training_csv_2[num2], 1))
    num2+=1
print("last_training_list", last_training_list_1,type(last_training_list_1),len(last_training_list_1))
goal1_mean=np.mean(last_training_list_1, axis=0)
goal2_mean=np.mean(last_training_list_2, axis=0)
print("goal1_mean,goal2_mean",goal1_mean,goal2_mean)


