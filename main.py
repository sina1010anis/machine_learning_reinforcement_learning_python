import numpy as np
import pandas as pd
#---------------START Section 2 ML------------------------
# np_arr = np.array([[1, 2], [3, 4]])

# np_mat = np.matrix([[1, 2], [3, 4]])

# mul_normal = np_arr@np_arr

# mul_num = np.dot(np_arr, np_arr);

# mul_num_mul = np.multiply(np_arr, np_arr);

# mul_num_pro = np.prod(np_arr);

# print(mul_num_pro)

#---------------END Section 2 ML------------------------

#---------------START Section 3 ML------------------------
    
# a = np.ones((3, 1))

# b = np.array([2, 6, 7])

# c = a+b

# print(c)

#---------------END Section 3 ML------------------------

#---------------START Section 4 ML------------------------
    
# np_arr = np.array([[1, 2], [3, 4]])

# sum = np.sum(np_arr)

# cumsum = np.cumsum(np_arr)

# rand = np.random.uniform(1, 5, (2, 3))

# rand_normal = np.random.standard_normal((2, 3))

# print(rand_normal)

#---------------END Section 4 ML------------------------


#---------------START Section 5 ML------------------------د

# s = np.ones((5, 9))

# np_arr = np.array([[1, 2], [3, 4]])

# print(np.shape(np_arr))

#---------------END Section 5 ML------------------------


#---------------START Section 6 ML------------------------

# a = np.array([1,8,7,5,6,8,7,3,1])

# b = np.array([9,4,2,3,6,4,8,1,2])

# c = np.array([18,19,20])

# m = np.array([1,2,2])

# a_unique = np.unique(a)

# a_union1d = np.union1d(a, b)

# a_inst = np.intersect1d(a, b)

# a_inst = np.intersect1d(a, b)

# c_mean = np.mean(c)

# c_meadian = np.median(c)

# c_std = np.std(c)

# c_var = np.var(c)

# m_polval = np.polyval(m, 1)

# m_polder = np.polyder(m)

# m_polint = np.polyint(m)

# print(m_polint)

#---------------END Section 6 ML------------------------


#---------------START Section 7 ML------------------------

# s = pd.Series([1,2,3,4], index=['row1', 'row2', 'row3', 'row4'])

# value = s.values

# index = s.index

# s.rename({'row1' : 0, 'row2' : 1, 'row3' : 2, 'row4' : 3})

# print(s)

#---------------START Section 7 ML------------------------


#---------------START Section 8 ML------------------------

DF = pd.DataFrame(np.array([[1 ,2 , 3], [4, 5 , 6]]), index=['row_1', 'row_2'], columns=['col_1', 'col_2', 'col_3'])

print(DF)

#---------------START Section 8 ML------------------------