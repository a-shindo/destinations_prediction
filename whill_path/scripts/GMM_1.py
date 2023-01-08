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
data = pd.read_csv('/home/ytpc2017d/catkin_ws/src/whill_path/scripts//total_dining_hall.csv')
X_train = data


# N = len(X_train)
# 2つのコンポーネントを持つガウス混合モデルに適合
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# モデルによる予測スコアを等高線図として表示します
x = np.linspace(-4, 6)
y = np.linspace(-16, 16)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)
plt.figure(figsize=[10,32])
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))# norm=LogNorm(vmin=1.0, vmax=1000.0),
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], .8)

# plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
