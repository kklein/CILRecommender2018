import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# df = pd.read_csv('../data/overall_scores.csv', dtype=np.float64)
df = pd.read_csv('overall_scores.csv', header=None, dtype={1: np.float64, 2: np.float64})

plt.scatter(df[0], df[1])
plt.scatter(df[0], df[2])
plt.show()
