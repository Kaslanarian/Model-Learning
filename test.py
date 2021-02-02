# %%
import numpy as np
import pandas as pd
import seaborn as sns

sns.get_dataset_names()
# %%
data = sns.load_dataset("iris")[:500]
data.to_excel("iris.xlsx")
data
# %%
d = dict(zip(list(set(data.species)), range(1, 4)))
data.species=[d[x] for x in data.species]
# %%
data
# %%
data.to_excel("iris.xlsx")

# %%
