import os
import glob 
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math

""""
https://qiita.com/minamini1985/items/60047ea416cd49721cf3
"""

# パスで指定したファイルの一覧をリスト形式で取得. （ここでは一階層下のtestファイル以下）
csv_files = sorted(glob.glob(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/csv_3/stairs/df_3/*.csv'))
# csv_files = sorted(glob.glob(f'/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2_downsampler_test/*.csv'))


#読み込むファイルのリストを表示
for a in csv_files:
    print(a)

#csvファイルの中身を追加していくリストを用意
data_list = []

#読み込むファイルのリストを走査
for file in csv_files:
    data_list.append(pd.read_csv(file))

#リストを全て行方向に結合
#axis=0:行方向に結合, sort
df = pd.concat(data_list, axis=0, sort=True)

df.to_csv(f"/home/ytpc2017d/catkin_ws/src/whill_path/scripts/csv_3/total_stairs/total_stairs_df_3.csv",index=False)

