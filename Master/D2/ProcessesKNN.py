# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import BUSC


def benchmark():

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D2_normal.csv')

    knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=1)

    knn.fit(x_tr, l_tr)

    l_pre = knn.predict(x_te)

    BUSC.score(knn, x_te, l_te, l_pre)


def showPlot():

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D2_normal.csv')

    knn = KNeighborsClassifier(n_neighbors=4)

    knn.fit(x_tr, l_tr)

    l_pre = knn.predict(x_te)

    return BUSC.score(knn, x_te, l_te, l_pre, mode='return')



# benchmark()



