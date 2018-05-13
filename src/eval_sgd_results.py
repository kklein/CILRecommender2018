import pandas as pd

df = pd.read_csv('../data/sgd_scores.csv', header=None, dtype=np.float64)
for k in [5 * i for i in range(1,21)]:
    mask = df[0] == k
    plt.plot(df[mask][1], df[mask][2])
plt.show()
for Lambda in [0, 0.005, 0.001, 0.0015, 0.002, 0.0025]:
    mask = df[1] == Lambda
    plt.scatter(df[mask][0], df[mask][2])
plt.show()
