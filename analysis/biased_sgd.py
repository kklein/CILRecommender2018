import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

df = pd.read_csv('biased_sgd_scores.csv', header=None,
        dtype=np.float64)

# df = df.sort_values(by=0)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_title("biased reg svd: validation RSME",fontsize=14)
ax.set_xlabel("#features",fontsize=12)
ax.set_ylabel("regularization",fontsize=12)
ax.grid(True,linestyle='-',color='0.75')

# scatter with colormap mapping to z value
ax.scatter(df[0],df[1],s=20,c=df[2], marker='o', cmap=cm.jet)

plt.show()

# plot_colourline(df[0], df[2], df[1])
# plt.show()

# plot_colourline(df[0], df[1], df[2])
# plt.show()
#
# df = df.sort_values(by=1)
#
# plot_colourline(df[0], df[1], df[2])
# plt.show()
#
# df = df.sort_values(by=2)
#
# plot_colourline(df[0], df[1], df[2])
# plt.show()

# x = df[0]
# y = df[2]
# plt.scatter(x, y)
# plt.show()
#
# x = df[1]
# plt.scatter(x, y)
# plt.show()
