from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

path_file = 'CatchPhish_D1.csv'

data = pd.read_csv(path_file)

data.rename(index=data.URL, inplace=True)

data.drop(['URL'], axis=1, inplace=True)

data.replace('?', np.nan)

data.fillna(method='ffill')

data.drop_duplicates()

data_normal = minmax_scale(data, feature_range=(0, 1))

data = pd.DataFrame(data_normal, columns=data.columns, index=data.index)

data.to_csv(path_file)

print(data)