# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import BUSC

x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv')

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(x_tr, l_tr)

l_pre = knn.predict(x_te)

BUSC.score(knn, x_te, l_te, l_pre)







