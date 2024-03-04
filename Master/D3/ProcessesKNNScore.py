import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_auc_score

data = pd.read_csv('CatchPhish_D3_normal.csv')

data.rename(index=data.URL, inplace=True)

labels = np.array(data.label)

data.drop(['label', 'URL'], axis=1, inplace=True)

data = np.array(data)

x_tr, x_te, l_tr, l_te = train_test_split(data, labels, test_size=0.3, random_state=True)

score = []

score_print = []

n = np.arange(30)


for i in n:

    knn = KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(x_tr, l_tr)

    score.append(knn.score(x_te, l_te))

    score_print.append(knn.score(x_te, l_te))

    score_print.append(i+1)

    print(i)

print(score_print)

plt.plot(n, score)

plt.xlabel('K')

plt.text(5, 0.8767449238578681, 'Best score')

plt.grid()

plt.legend(['Score'], loc='best')

plt.scatter(4, 0.8767449238578681, marker='o', c='red', s=120)

plt.title('KNN Best K')

plt.show()