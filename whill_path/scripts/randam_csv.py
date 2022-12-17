from glob import glob
from os.path import join
import random
import pandas as pd

"""
https://vasteelab.com/2018/06/03/2018-06-03-141238/
https://biotech-lab.org/articles/10669

"""

csv_foldar = "/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal1/"
# 指定パスのファイルのリストを取得
files = glob(join(csv_foldar, "*.csv"))
# ファイルの総数を取得
num_files = len(files)
# ファイルのリストを3:20に分ける
test = random.sample(files, int(num_files*(3/23)))
training = random.sample(files, num_files - int(num_files*(3/23)))

print(test[1])
 
df = pd.read_csv(test[1])
for index, data in df.iterrows():
    print(index)
    print(data)
    print('--------')

print(test[1])