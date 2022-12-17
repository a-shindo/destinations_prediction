# import glob
# import random
# import os 
# import shutil
# import math

# INPUT_DIR = '/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2/'
# OUTPUT_DIR = 'out'
# #ランダムで抽出する割合
# SAMPLING_RATIO = 3/23

# def random_sample_file():
#     files = glob.glob(INPUT_DIR + '*.csv')

#     random_sample_file = random.sample(files,math.ceil(len(files)*SAMPLING_RATIO))
#     os.makedirs(OUTPUT_DIR + "/choices",exist_ok=True)

#     for file in random_sample_file:
#         shutil.copy2(file,OUTPUT_DIR + "/choices/")

# if __name__ == '__main__':
#     random_sample_file()












# import numpy as np
# import pandas as pd
# from glob import glob
# import pandas as pd
# from sklearn.model_selection import KFold #KFoldのインポート

# csv_files = glob("/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2//*.csv")
# df = pd.concat([pd.read_csv(i) for i in csv_files])
# print("1111111111111111111111111111", df, "type(df)", type(df))
# import glob
# df2 = [glob.glob("/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2//*.csv")]
# print("22222222222222222222222222", df2, "type(df2)", type(df2))

# # #データの準備
# # all = dict(
# #     #分割する文字列
# #     data = ["0","1","2","3","4","5","6","7","8","9"],
# #     #目的変数
# #     label = [1,1,1,1,1,0,0,0,0,0],
# #     #dataが所属するグループ
# #     group = [1,0,1,0,1,0,1,0,1,0],
# #     ) 
# # #DataFrameを作成
# # df2 = pd.DataFrame(data = all)
# # print("22222222222222222222222222", df2, "type(df2)", type(df2))
# #各列を抽出


# # data = df["data"]
# # label = df["label"]
# # group = df["group"]


 
# #KFoldの設定
# kf = KFold(n_splits = 5, shuffle = True, random_state = 3)
 
# #交差検証
# for train, test in kf.split(df2):
#   print(f"訓練データは：{df2[train]}, テストデータは：{df2[test].values}")
#   # print(f"訓練ラベルは：{label[train].values}, テストラベルは：{label[test].values}")

# # from glob import glob
# # from os.path import join
# # import random

# # csvPath='/home/ytpc2017d/catkin_ws/src/whill_path/scripts/goal2/'
# # files = glob(join(csvPath, "*.csv"))
# # # ファイルの総数を取得
# # num_files = len(files)
# # # ファイルのリストを7:3に分ける
# # files30 = random.sample(files, int(num_files*0.3))
# # files70 = random.sample(files, num_files - int(num_files*0.3))