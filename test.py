# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_excel('iris.xlsx').drop(axis=1, columns='Unnamed: 0')

# sns.set()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = data['petal_length'], data['sepal_length'], data['sepal_width']

ax.set_xlabel("petal_length")
ax.set_ylabel("sepal_length")
ax.set_zlabel("sepal_width")

ax.scatter(x, y, z, c=data.species)

plt.savefig('iris_species_scatter.png')

# %%
