# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import BUSC


def benchmark():
    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D2_normal.csv')

    # x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv', ['amp_greater_equal', 'delims_url', 'len_url', 'email_exist', 'protocol_url', 'digits_url', 'brand_host', 'host_large_tok', 'len_path', 'digits_path'])

    nb = GaussianNB(var_smoothing=1e-01)

    nb.fit(x_tr, l_tr)

    l_pre = nb.predict(x_te)

    BUSC.score(nb, x_te, l_te, l_pre)

def showPlot():

    x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D2_normal.csv')

    # x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv', ['amp_greater_equal', 'delims_url', 'len_url', 'email_exist', 'protocol_url', 'digits_url', 'brand_host', 'host_large_tok', 'len_path', 'digits_path'])

    nb = GaussianNB(var_smoothing=1e-01)

    nb.fit(x_tr, l_tr)

    l_pre = nb.predict(x_te)

    return BUSC.score(nb, x_te, l_te, l_pre, mode='return')

# benchmark()
