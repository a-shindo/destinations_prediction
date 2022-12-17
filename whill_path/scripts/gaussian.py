import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
# from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

"""
http://int-info.com/PyLearn/PyLearnPR03.html
"""

data = pd.read_csv("./total1.csv")
x=data["x"]
y=data["y"]

mean_xy = np.mean(data, 0)
cov_xy = np.cov(data, rowvar=False)
print("mean_x, mean_y", mean_xy)
print("cov_xy", cov_xy)
# Sigma11, Sigma12, Sigma21, Sigma22 = cov_xy.reshape(-1)
X = np.linspace(np.min(x)-1,np.max(x)+1)
Y = np.linspace(np.min(y)-1,np.max(y)+1)
XX, YY = np.meshgrid(X,Y)
z = np.dstack((XX, YY))
pdf2 = multivariate_normal.pdf(z, mean_xy, cov_xy)
print("pdf2", pdf2)



plt.contour(XX, YY, pdf2, cmap='PuOr')
# plt.pcolor(XX, YY, pdf2, cmap="hot")
plt.colorbar() # カラーバー
# plt.title('visualization', size=20)
plt.scatter(data["x"], data["y"], s=4, c="gray")
plt.xlabel('x', size=20)
plt.ylabel('y', size=20)
plt.show()








