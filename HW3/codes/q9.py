import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
### Part A
df = pd.read_csv("Iris.csv", index_col=False)

for l1, l2 in combinations(df.columns[:-1]):
    





# encoder = LabelEncoder()
# y = encoder.fit_transform(df["Class"])
# X = df["Sepal_Length,Sepal_Width,Petal_Length,Petal_Width".split(',')].to_numpy()

