from inspect import isclass
from os import close
from mpmath.visualization import plot
import numpy as np
from sympy.plotting.plot import plot_parametric
from sympy.solvers import solve, solveset
from sympy import Symbol, lambdify
import matplotlib.pyplot as plt
from sympy.plotting import plot3d, plot


X = [
    [-10, -10, -8, -5, -4, -2, 0, 2, 3, 4, 5, 5, 8],
    [-6, -5, -3, -2, -3, 1, 1, 4, 4, 3, 3, 5, 9]
]

labels = [0] * 7 + [1] * 6

Y = np.array(labels)

X = np.array(X)


x_red = X[...,Y==1]
x_yellow = X[...,Y==0]


mr = x_red.mean(axis=1, keepdims=True)
sigma_r = np.cov(x_red)

my = x_yellow.mean(axis=1, keepdims=True)
sigma_y = np.cov(x_yellow)

print(mr)

x1 = Symbol('x')
x2 = Symbol('y')

x = np.array([x1, x2]).reshape(-1, 1)


fy = -1/2*(x-my).T @ np.linalg.inv(sigma_y) @ (x-my)
fr = -1/2*(x-mr).T @ np.linalg.inv(sigma_r) @ (x-mr)

fy, fr = fy[0,0], fr[0,0]


coef = np.log(6/7 * np.sqrt(np.linalg.det(sigma_y)/np.linalg.det(sigma_r)))

eq1, eq2 = solve(fy-fr+coef, [x1,x2])

print(eq1)
print(solve(eq1[0]))

X1 = np.linspace(X[0,:].min(), X[0,:].max(), 50)
X2 = np.linspace(X[1,:].min(), X[1,:].max(), 50)


f1 = lambdify([x1, x2], eq1[0])
f2 = lambdify([x1, x2], eq2[0])

V1 = np.array([f1(0, v2) for v2 in X2])
V2 = np.array([f2(0, v2) for v2 in X2])

fig,ax = plt.subplots()

ax.plot(V1, X2, label="eq1", c="purple")
ax.plot(V2, X2, label="eq2", c="navy")
ax.scatter(X[0, :], X[1, :], c=Y)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xlim(X[0, :].min()-5, X[0, :].max() + 5)
ax.set_ylim(X[1, :].min()-5, X[1, :].max() + 5)
ax.grid()
plt.show()