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

data1, data2 = pd.read_csv("./total1.csv"), pd.read_csv("./total2.csv")

mean_xy_1, mean_xy_2 = np.mean(data1, 0),np.mean(data2, 0)
cov_xy_1 = np.cov(data1, rowvar=False)
cov_xy_2 = np.cov(data2, rowvar=False)
print("mean_xy_1, mean_xy_2 ", mean_xy_1, mean_xy_2 )
print("cov_xy", cov_xy_1,cov_xy_2)
# Sigma11, Sigma12, Sigma21, Sigma22 = cov_xy.reshape(-1)
X_1 = np.linspace(np.min(data1["x"])-1,np.max(data1["x"])+1)
Y_1 = np.linspace(np.min(data1["y"])-1,np.max(data1["y"])+1)
XX_1, YY_1 = np.meshgrid(X_1,Y_1)
z_1 = np.dstack((XX_1, YY_1))
pdf1 = multivariate_normal.pdf(z_1, mean_xy_1, cov_xy_1)
print("pdf1", pdf1)

X_2 = np.linspace(np.min(data2["x"])-1,np.max(data2["x"])+1)
Y_2 = np.linspace(np.min(data2["y"])-1,np.max(data2["y"])+1)
XX_2, YY_2 = np.meshgrid(X_2,Y_2)
z_2 = np.dstack((XX_2, YY_2))
pdf2 = multivariate_normal.pdf(z_2, mean_xy_2, cov_xy_2)
print("pdf2", pdf2)

plt.figure(figsize=[6,10])
plt.xlim(1, 7)
plt.ylim(1, 13)
plt.contour(XX_1, YY_1, pdf1, cmap='Blues')
plt.colorbar() # カラーバー
plt.contour(XX_2, YY_2, pdf2, cmap='Reds')
plt.colorbar() # カラーバー
plt.scatter(data1["x"], data1["y"], s=4, c="lightblue")
plt.scatter(data2["x"], data2["y"], s=4, c="pink")
# plt.title('visualization', size=20)
plt.xlabel('x', size=20)
plt.ylabel('y', size=20)
plt.show()
# shape = XX.shape
# XX = XX.reshape(-1)
# YY = YY.reshape(-1)

# plt.title("multivariate gaussian")
# plt.contourf(multivariate_normal.pdf(data, 
#                                     mean=[mean_xy], 
#                                     cov=cov_xy))
# # plt.contourf(XX.reshape(shape), YY.reshape(shape), 
# #             multivariate_normal.pdf(np.array(list(zip(XX, YY))), 
# #                                     mean=[mean_xy], 
# #                                     cov=cov_xy).reshape(shape))
# plt.scatter(x, y, 
#             alpha=0.3, linewidths=0, marker=".", c="gray")









