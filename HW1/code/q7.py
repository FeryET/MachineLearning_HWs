import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MinMaxScaler, PolynomialFeatures,
                                   StandardScaler)
from tqdm import tqdm

import initializer

sns.set_theme()
sns.set_context("talk")


student_number = "810199091"

sigma = int(student_number[::-1][1]) ** 1/3

df = pd.read_csv("data/poly.csv")

X, y =np.array(df["X"], dtype=np.float64), np.array(df["Y"], dtype=np.float64)

X += np.random.normal(loc=1, scale=sigma, size=X.shape)
y += np.random.normal(loc=1, scale=sigma, size=y.shape)

X_plot = X.copy()

X = X.reshape(-1, 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

labels = {0: "original"}

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15), sharex=True, sharey=True)


x_range_plot = np.arange(X_plot.min()-100, X_plot.max()+100).reshape(-1,1)
x_range = scaler.transform(x_range_plot)

degs = [1, 3, 7, 11, 16, 20]
info = []

num_out = 1
print("training starts..")
for d, ax in zip(degs, axes.flat):
    polyreg = make_pipeline(PolynomialFeatures(d,),LinearRegression())
    polyreg.fit(X, y)
    pred = polyreg.predict(X)
    regressed_region = polyreg.predict(x_range)
    bias = np.sqrt(((pred - y) ** 2).mean())
    var = np.var(pred)
    ax.plot(x_range_plot, regressed_region, label=f"degree={d}", c="b")#, linewidth=0.7)
    ax.scatter(X_plot, y, label="original", s=5, c="r")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Polynomial Degree = {d}")
    info.append({"degree": d, 
                "mse": mean_squared_error(y, pred),
                "bias": bias,
                "variance": var})

axes.flat[0].set_ylim(bottom=y.min() - 20, top=(y.max() + 20))
fig.suptitle("Regression Results for Different Polynomial Degrees")

# axes.flat[0].set_ylim(bottom=(y.min() ** 5)//10, top=(y.max() ** 105)//100)

# plt.tight_layout()
plt.savefig("outputs/q7_regression.png")

df = pd.DataFrame(info)

df.to_csv("outputs/q7_info.csv")

bias = np.array(df["bias"])
var = np.array(df["variance"])
loss = np.array(df["mse"])

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9,9))
axes.flat[0].plot(degs, bias)
axes.flat[0].set_ylabel("Bias")
axes.flat[1].plot(degs, var)
axes.flat[1].set_ylabel("Variance")
axes.flat[2].plot(degs, loss)
axes.flat[2].set_ylabel("Loss")
axes.flat[2].set_xlabel("Polynomial Degree")


fig.suptitle("Polynomial Complexity Effects")
plt.tight_layout()
plt.savefig("outputs/q7_complexity_plot.png")
