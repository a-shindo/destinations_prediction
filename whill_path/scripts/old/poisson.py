import numpy as np
from scipy import stats
from tqdm import tqdm

import matplotlib.pyplot as plt
# %matplotlib inline

def func2(w, x):
    """パラメータが従う関数
    """
    return np.exp(w[0] + w[1] * x)

# *** データを作ってみる
# 真の関数(これを推論したい)
w_true = [-0.0, 0.3]

# サンプルデータ
N = 100
x_min, x_max = 3, 10
X = stats.uniform(loc=x_min, scale=(x_max-x_min)).rvs(N)
y = np.array([stats.poisson(mu=func2(w_true, x)).rvs() for x in X])

# # 作ったサンプルデータと真の関数を可視化する
# fig = plt.figure(figsize=(8, 4))
# ax = fig.subplots(1,1)

# x = np.linspace(x_min, x_max, 100)
# lam_true = func2(w_true, x)
# ax.scatter(x, lam_true, color='red', alpha=0.5, label='TrueFunction')
# ax.scatter(X, y, color='blue', alpha=0.5, label='ToyData')
# ax.set_xlabel('x')
# ax.set_ylabel('$\\mu$')
# ax.legend()

def err_func(lam, y):
    """対数尤度の符号反転
    対数尤度に含まれる$\ln y!$の項は、muに無関係のため省略。
    """
    loss = - (y * np.log(lam) - lam).sum()
    return loss

# %time
# *** パラメータbetaを推論する
lr = 0.00001
w = np.array([0.,0.])
loss = []
n_iter = 100
for i in tqdm(range(n_iter)):
    lambda_k = func2(w, X)
    grad = np.array(
        [np.sum(lambda_k - y), 
         np.sum((lambda_k - y) * X)])
    loss.append(err_func(lambda_k, y))
    w[0] = w[0] - lr * grad[0]
    w[1] = w[1] - lr * grad[1]

print(f'estimated_parameter : {w}')

# *** lossの値を確認(学習曲線)
fig = plt.figure(figsize=(8, 4))
ax = fig.subplots(1,1)

ax.plot(loss)

# *** 推論結果をplot
fig = plt.figure(figsize=(8, 4))
ax = fig.subplots(1,1)

x = np.linspace(x_min, x_max, 100)
lam_true = func2(w_true, x)
ax.scatter(x, lam_true, color='red', alpha=0.5, label='TrueFunction')
ax.scatter(X, y, color='blue', alpha=0.5, label='ToyData')
ax.set_xlabel('x')
ax.set_ylabel('$\\theta$')

lam_mle = func2(w, x)
ax.scatter(x, lam_mle, color='green', alpha=0.5, label='EstFunc(MLE)')

ax.legend()



plt.show()