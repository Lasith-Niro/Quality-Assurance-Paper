import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
df = df.iloc[:, 1:]

df = df.drop('hasHoles', axis=1)
df.head()

corr_matrix = np.corrcoef(df).round(decimals=2)
df.corr(method='spearman')

sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=False, cmap="rocket_r")
plt.show()