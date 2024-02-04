import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#---------------START Section 2 ML------------------------
# np_arr = np.array([[1, 2], [3, 4]])

# np_mat = np.matrix([[1, 2], [3, 4]])

# mul_normal = np_arr@np_arr # ضرب ماتریسی

# mul_num = np.dot(np_arr, np_arr); # ضرب ماتریسی با NUMPY

# mul_num_mul = np.multiply(np_arr, np_arr); # ضرب ماتریسی هر سلول به سلول متازرNUMPY

# mul_num_pro = np.prod(np_arr); # ضرب هر سلول با سلول بعدی در یک ماتریس

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

# cumsum = np.cumsum(np_arr) # جمع هر سلول با سلول بعدی در یک ماتریس

# subtract = np.subtract(np_arr, np_arr) # کم کردن دو ارتیس از هم

# div = np.divide(np_arr, np_arr) # تقسیم کردن دو ارتیس از هم

# div_f = np.floor_divide(np_arr, np_arr) # تقسیم کردن دو ارتیس از هم با نمایش ندادن اعشار

# sqrt = np.sqrt(np_arr) # جذر گرفتن از یک ماتریس یا هر عددی

# rand = np.random.uniform(1, 5, (2, 3)) # ساخت یک ماتریس با دو سطر و سه ستون که با نقدار های 1 تا 5 پر شده انند

# rand_normal = np.random.standard_normal((2, 3)) # مثل بالای فقط به جای عدد بین یک عدد نرمال میزاره

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

# a_unique = np.unique(a) # پیدا کردن عدد های تک در ناتریس

# a_union1d = np.union1d(a, b) # پیدا کردن اتخاد بیندو ماتریس

# a_inst = np.intersect1d(a, b)# پیدا کردن اشتراک بیندو ماتریس

# a_inst = np.intersect1d(a, b)

# c_mean = np.mean(c) # مایانگین یک ماتریس

# c_meadian = np.median(c)  # median یک ماتریس

# c_std = np.std(c)  # str یک ماتریس

# c_var = np.var(c)  # واریانس یک ماتریس

# m_polval = np.polyval(m, 1)  # مشخص کردن x ها در معادله چندجملهای

# m_polder = np.polyder(m) # مشتق گرفتن از چند جمالهای 

# m_polint = np.polyint(m) # انتگرال گرفتن از چند جمله ای

# print(m_polint)

#---------------END Section 6 ML------------------------


#---------------START Section 7 ML------------------------

# s = pd.Series([1,2,3,4], index=['row1', 'row2', 'row3', 'row4']) # شاخت یک نوع داده یک بعدی یا Seres

# value = s.values # گرفتن مقدار های یک نوع داده

# index = s.index # گرفتن نام برچسب های یک داده

# s.rename({'row1' : 0, 'row2' : 1, 'row3' : 2, 'row4' : 3}) # تغییر نام برچسب ها

# print(s)

#---------------END Section 7 ML------------------------


#---------------START Section 8 ML------------------------

# DF = pd.DataFrame(np.array([[1 ,2 , 3], [4, 5 , 6]]), index=['row_1', 'row_2'], columns=['col_1', 'col_2', 'col_3']) # شاخت یک نوع داده دو بعدی یا Data frame

# print(DF)

#---------------END Section 8 ML------------------------


#---------------START Section 9 ML------------------------

# DF = pd.DataFrame(np.array([[1 ,2 , 3], [4, 5 , 6]]), index=['row_1', 'row_2'], columns=['col_1', 'col_2', 'col_3']) # شاخت یک نوع داده دو بعدی یا Data frame

# index = DF.index # گرفتم مقدار index

# col = DF.columns # گرفتن مقدار col

# values = DF.values # گرفتن مقدار values

# getForName = DF.loc['row_1']['col_2']# گرفتم داده از دیتافریم با نام ها

# getForNumber = DF.iloc[1][2] # گرفتم داده از دیتافریم با عدد ها

# DF['col_4'] = [98, 7] # اضافه کردن مقدار جدید به دیتافریم

# DF.drop('col_4', axis=1, inplace=True) # حذف col مورد نظر (مقدار axis برای این است که قرار col حذف به نه index اگر index بود باید 0 بزاریم) (مقدار inplace برای این است داخل خودش ذخیره بشه و سیو بشه اگر فالس بشه باید دخال یک متغیر دیگه ریخته بشه)

# DF.rename(columns={'col_1' : 'c_1'}, inplace=True)

# DF.replace(2, 0)

# print(DF)

#---------------END Section 9 ML------------------------


#---------------START Section 10 ML------------------------

# DF = pd.DataFrame(np.array([[1 ,8 , 6], [4, 5 , 3]]), index=['row_1', 'row_2'], columns=['col_1', 'col_2', 'col_3']) # شاخت یک نوع داده دو بعدی یا Data frame

# DF['col_1'] = DF['col_1'].apply(lambda x: x*2) # تغییر داده ها داخل دیتافریم

# DF.sort_values(by='col_1', ascending=False) # مرتب سازی داده ها بر اساسا نام col

# DF_head = DF.head(1) # نمایش از یک سطر از داده از بالا

# DF_tail = DF.tail(1) # نمایش از یک سطر از داده از پایین

# print(DF_tail)
#---------------END Section 10 ML------------------------


#---------------START Section 11 ML------------------------

plt.plot([20, 98, 5], [10, 5, 20]) #  خطی ساخت نمودار

plt.scatter([20, 98, 5], [10, 5, 20]) #  خطی ساخت نمودار

plt.show() # نمایش نموادر

#---------------END Section 11 ML------------------------
