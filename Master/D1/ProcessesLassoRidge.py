# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
import BUSC

x_tr, x_te, l_tr, l_te, data_x, labels, data = BUSC.buildData('CatchPhish_D1_normal.csv')

data = pd.DataFrame(data_x, columns=data.columns, index=data.index)

ri = Lasso(alpha=0.1)

ri.fit(x_tr, l_tr)

l_pre = ri.predict(x_te)

c = 0

fetrue_drop = []

for i in range(len(ri.coef_)):

    if ri.coef_[i] == 0:
        fetrue_drop.append(data.columns[i])

# Ridge ['amp_greater_equal', 'delims_url', 'len_url', 'email_exist', 'protocol_url', 'digits_url', 'brand_host', 'host_large_tok', 'len_path', 'digits_path']
        
# Ridge ['at_url', 'amp_greater_equal', 'delims_url', 'other_delims_url', 'len_url', 'email_exist', 'protocol_url', 'suspwords_url', 'tiny_url', 'digits_url', 'entropy_url', 'dot_host', 'len_subdomain', 'having_https', 'brand_host', 'host_large_tok', 'path_large_tok', 'dot_path', 'slash_path', 'len_path', 'brand_path', 'digits_path', 'len_file', 'extension', 'delims_params', 'len_params', 'ratio_url_path']

print(fetrue_drop)

# data.drop(fetrue_drop, axis=1, inplace=True)

# print(data)