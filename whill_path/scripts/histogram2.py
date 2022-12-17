import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

"""
https://qiita.com/supersaiakujin/items/ca47200393180a693bdf
"""

data = pd.read_csv("./total1.csv")
x=data["x"]
y=data["y"]

xx,yy = np.mgrid[1:7:0.1,7:13:0.1]
positions = np.vstack([xx.ravel(),yy.ravel()])

value = np.vstack([x,y])

kernel = gaussian_kde(value)

f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.contourf(xx,yy,f, cmap=cm.jet)
# ax.set_title('11th graph')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()










