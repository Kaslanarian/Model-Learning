# %%
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_excel('iris.xlsx').drop(axis=1, columns='Unnamed: 0')
sns.regplot(data=data, x='sepal_length', y='petal_width', logx=True)
data

# %%
sns.regression.algo.bootstrap(data)
# %%
data.plot()
# %%
data.to_numpy()
# %%
from sklearn import linear_model
from sklearn.metrics import r2_score
data_copy = data.copy()
model = linear_model.LinearRegression()
model.fit(X=data.to_numpy()[:, [0, 1, 2, 4]], y=data.petal_width)
corf = model.coef_
score = model.score(X=data.to_numpy()[:, [0, 1, 2, 4]], y=data.petal_width)
# %%
score
# %%
data.describe()
# %%
describe = data.describe()
data_copy = data.copy()
for col in data_copy.columns[:-1]:
    data_copy[col] = (data[col] - describe.loc['mean', col]) / describe.loc['std', col]
model.fit(X=data_copy.to_numpy()[:, [0, 1, 2, 4]], y=data_copy.petal_width)
model.coef_
# %%
data_copy.to_excel('standard_diamonds.xlsx')
# %%
