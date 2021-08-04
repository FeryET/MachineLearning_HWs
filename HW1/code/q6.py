from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, recall_score, precision_score, accuracy_score

dataset_path = "data/Q6_Dataset/Images"

class DatasetReader:
        
    def __init__(self, path, return_as_dict=False):
        self._path = path
        self._files = []
        for root, __, files in os.walk(self._path):
            for f in files:
                self._files.append(os.path.join(root, f))
        self.return_as_dict = return_as_dict

    def __next__(self):
        if self._index == self.__len__():
            raise StopIteration
        fpath = Path(self._files[self._index])
        img = Image.open(fpath)
        img = np.array(img)
        label = fpath.name[0].lower()
        self._index += 1
        if self.return_as_dict:
            return {"mean": img.mean(axis=(0,1)), "label": label}
        else:
            return img.mean(), label

    def __iter__(self):
        self._index = 0        
        return self

    def __len__(self):
        return len(self._files)


dataset = DatasetReader(dataset_path, return_as_dict=True)
# print(len(dataset))
dataset = tqdm(dataset, total=len(dataset))
dataset = list(dataset)
df = pd.DataFrame(dataset)

df = df[(df["label"] == "c") | (df["label"] == "m")]
classes = {"m": "ManUtd", "c": "Chelsea"}
df["label"] = df["label"].apply(lambda x: classes[x])
encoder = LabelEncoder()
df["encoded"] = encoder.fit_transform(df["label"])

X = list(df["mean"])
X = np.stack( X, axis=0 )
y = np.array(df["encoded"])
indices = y == 0
colors = {"ManUtd": "r", "Chelsea": "b"}

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,10))
int_colors = {idx: k for idx, k in enumerate(list("rgb"))}
for idx, ax in enumerate(axes):
  c = np.array([colors[encoder.classes_[0]]] * len(indices))
  c[~indices] = colors[encoder.classes_[1]]
  ax.hist([X[indices, idx], X[~indices, idx]], bins=50, label=encoder.classes_)
  ax.set_ylabel(f"{int_colors[idx].upper()} Intensities")

fig.suptitle("Histogram of Average Pixel Value for Different Colors")
axes[0].legend()
plt.savefig("outputs/q6_hist.png")
plt.close()

df[["r", "g", "b"]] = list(X)
fig,axes = plt.subplots(nrows=3, figsize=(10,10))
for x, ax in zip("rgb", axes):
  df.loc[indices][x].plot(kind='density', label=encoder.classes_[0], c=colors[encoder.classes_[0]], ax=ax)
  df.loc[~indices][x].plot(kind='density', label=encoder.classes_[1], c=colors[encoder.classes_[1]], ax=ax)
  ax.set_ylabel(f"Density")
  ax.set_xlabel(f"Average Channel {x.upper()} Pixel Intensity")
axes[0].legend()
plt.savefig("outputs/q6_desnity_plot.png")


clf = KNeighborsClassifier(n_neighbors=5)

clf.fit(X, y)

prediction = clf.predict(X)

# y = encoder.inverse_transform(y)
# prediction = encoder.inverse_transform(prediction)

conf_mat = confusion_matrix(y, prediction)
fig, ax = plt.subplots()
plot_confusion_matrix(clf, X, y, ax=ax, display_labels=encoder.classes_)
plt.savefig("outputs/q6_confusion_matrix.png")

info = {
    "recall": recall_score(y, prediction),
    "precision": precision_score(y, prediction),
    "accuracy": accuracy_score(y, prediction)
}

import json

with open("outputs/q6_info.json", "w") as jsfile:
    json.dump(info, jsfile)
