# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import BUSC


def benchmark():

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv')

    rf = RandomForestClassifier(n_estimators=150, max_leaf_nodes=3000)

    print('n_estimators=150, max_leaf_nodes=3000')

    rf.fit(x_tr, l_tr)

    l_pre = rf.predict(x_te)

    BUSC.score(rf, x_te, l_te, l_pre)


def showPlot():

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv')

    rf = RandomForestClassifier(n_estimators=200, max_leaf_nodes=4000)

    rf.fit(x_tr, l_tr)

    l_pre = rf.predict(x_te)

    return BUSC.score(rf, x_te, l_te, l_pre, mode='return')



# benchmark()




