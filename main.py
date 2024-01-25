import numpy as np
#---------------START Section 2 ML------------------------
np_arr = np.array([[1, 2], [3, 4]])

np_mat = np.matrix([[1, 2], [3, 4]])

mul_normal = np_arr@np_arr

mul_num = np.dot(np_arr, np_arr);

mul_num_mul = np.multiply(np_arr, np_arr);

mul_num_pro = np.prod(np_arr);

print(mul_num_pro)

#---------------END Section 2 ML------------------------

