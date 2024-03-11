import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
import BUSC

def benchmark():
    # x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D3_normal.csv')

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D3_normal.csv', ['amp_greater_equal', 'delims_url', 'len_url', 'email_exist', 'protocol_url', 'digits_url', 'digits_path'])
    LR = LogisticRegression()

    LR.fit(x_tr, l_tr)

    l_pre = LR.predict(x_te)

    BUSC.score(LR, x_te, l_te, l_pre)

def showPlot():
    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D3_normal.csv')

    # x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D3_normal.csv', ['amp_greater_equal', 'delims_url', 'len_url', 'email_exist', 'protocol_url', 'digits_url', 'digits_path'])

    LR = LogisticRegression()

    LR.fit(x_tr, l_tr)

    l_pre = LR.predict(x_te)

    return BUSC.score(LR, x_te, l_te, l_pre, mode='return')

# benchmark()