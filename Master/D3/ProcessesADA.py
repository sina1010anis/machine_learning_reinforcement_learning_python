# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import BUSC


def benchmark():

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv')

    ab = AdaBoostClassifier(n_estimators=150, learning_rate=0.5)

    print('n_estimators=150, learning_rate=0.5')

    ab.fit(x_tr, l_tr)

    l_pre = ab.predict(x_te)

    BUSC.score(ab, x_te, l_te, l_pre)


def showPlot():

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv')

    ab = AdaBoostClassifier(n_estimators=150, learning_rate=0.5)

    ab.fit(x_tr, l_tr)

    l_pre = ab.predict(x_te)

    return BUSC.score(ab, x_te, l_te, l_pre, mode='return')



# benchmark()




