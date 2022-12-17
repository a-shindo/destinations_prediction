import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm

"""
https://qiita.com/supersaiakujin/items/ca47200393180a693bdf
"""
data = pd.read_csv("./total1.csv")
x=data["x"]
y=data["y"]
fig = plt.figure()
ax = fig.add_subplot(111)

counts, xedges, yedges, Image = ax.hist2d(x,y, bins=[np.linspace(1,7,35),np.linspace(7,13,35)], norm=matplotlib.colors.LogNorm(), cmap=cm.jet)
ax.contour(counts.transpose(),extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()])
# H = ax.hist2d(x,y, bins=40, cmap=cm.jet)
# ax.set_title('1st graph')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(Image,ax=ax)
plt.show()


# kde = kale.KDE(data, reflect=[[0, None], [None, 1]])


# data.head()
# mean_x = np.mean(data["x"])
# mean_y = np.mean(data["y"])
# mean_xy =np.mean(data)
# cov_xy = np.cov(data, rowvar=0)
# Sigma11, Sigma12, Sigma21, Sigma22 = cov_xy.reshape(-1)
# print("mean_x, mean_y", mean_x, mean_y)
# print("cov_xy", cov_xy)

# kde = gaussian_kde(data)
# print(kde.evaluate([1, 1]))



# x = np.linspace(np.min(data["x"])-5,np.max(data["x"])+5)
# y = np.linspace(np.min(data["y"])-5,np.max(data["y"])+5)
# xx, yy = np.meshgrid(x,y)
# meshdata = np.vstack([xx.ravel(), yy.ravel()])
# z = kde.evaluate(meshdata)

# fig = plt.figure(facecolor="w")
# ax = fig.add_subplot(111, title="カーネル密度推定")
# ax.scatter(data[:, 0], data[:, 1], c="b")
# ax.contourf(xx, yy, z.reshape(len(y), len(x)), cmap="Blues", alpha=0.5)
# plt.show()












