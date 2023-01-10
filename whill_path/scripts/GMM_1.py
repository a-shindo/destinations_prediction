import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab


"""
https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py
"""

# データをロード
# data = pd.read_csv('/home/ytpc2017d/catkin_ws/src/whill_path/scripts//total_dining_hall.csv')
# data = pd.read_csv('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_elevator.csv')
# data = pd.read_csv('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_staff_station.csv')
data = pd.read_csv('/home/ytpc2017d/catkin_ws/src/whill_path/scripts/total_stairs.csv')
X_train = data


# N = len(X_train)
# 2つのコンポーネントを持つガウス混合モデルに適合
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

print("weights_", clf.weights_) # 各混合成分の重量
print("means_", clf.means_) # 各混合成分の平均
print("covariances_", clf.covariances_) # 各混合成分の共分散
print("precisions_", clf.precisions_) # 混合物中の各成分の精度行列(=共分散行列の逆行列)
print("precisions_cholesky_", clf.precisions_cholesky_) # 各混合成分の精度行列のコレスキー分解
print("converged_", clf.converged_) # 収束(True or False)
print("n_iter_", clf.n_iter_) # 収束に到達するために EM の最適適合で使用されるステップ数
print("lower_bound_", clf.lower_bound_) # EM の最適適合の (モデルに関するトレーニング データの) 対数尤度の下限値
print("n_features_in_", clf.n_features_in_) # フィット中に見られる特徴の数

# モデルによる予測スコアを等高線図として表示します
x = np.linspace(-4, 6)
y = np.linspace(-16, 16)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)
plt.figure(figsize=[10,32])
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10), cmap='Purples')# norm=LogNorm(vmin=1.0, vmax=1000.0),
# 'Oranges', 'YlGn', 'Blues', 'Purples'
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], .8, c="mediumpurple")
#'orange', 'yellowgreen', 'lightblue', 'mediumpurple',
# plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
