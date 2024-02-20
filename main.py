import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import pearsonr,chi2_contingency
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
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

# plt.plot([20, 98, 5], [10, 5, 20]) #  خطی ساخت نمودار

# plt.scatter([20, 98, 5], [10, 5, 20]) #  خطی ساخت نمودار

# plt.hist([25, 75, 10, 98], bins=2) # نمایش هیستوگررام

# plt.show() # نمایش نموادر

#---------------END Section 11 ML------------------------


#---------------START Section 12 ML------------------------

# plt.figure(figsize=(10, 10), dpi=100) # تغییر نمایشی نمودار الان در اینجا سایز پمچره نمودار 10 در 10 است و رزولوشن 100 است

# plt.plot([20, 98, 5], [10, 5, 20]) #  خطی ساخت نمودار

# plt.xlabel('Number 1') # روی محور x یک برچسب میزند

# plt.ylabel('Number 2') # روی محور y یک برچسب میزند

# plt.yticks([10, 5, 20], ['MLS', 'PWO', 'IPW']) # روی محور y مقدار های 10 و 5 20 را پیدا میکند و با مقدار های ارایه بعدی جای گذاری میکند

# plt.show() # نمایش نموادر

#---------------END Section 12 ML------------------------


#---------------START Section 13, 14, 15, 16 ML------------------------

# city_name = np.array(['X_C1', 'X_C2', 'X_C3', 'X_C4'])
# city_pop = np.array([878515214, 568514534, 321512584, 139614584]) / 1000000

# city_name_y = np.array(['Y_C1', 'Y_C2', 'Y_C3', 'Y_C4'])
# city_pop_y = np.array([448515214, 338514534, 221512584, 119614584]) / 1000000

# plt.figure(figsize=(10, 7)) # تغییر پنچره نمنایشی نمودار

# plt.subplot(1, 2, 1) # تقسیم بندی نمایش نمودار الان یک سط و دو ستون برای نمایش هست که بخش اول هست برای نمودار خط بعدی

# plt.plot(np.arange(len(city_name)), city_pop, ls='-' , marker='+', mew=5) # نمایش نمودار به صورت نقطه ای 

# plt.subplot(1, 2, 2) # تقسیم بندی نمایش نمودار الان یک سط و دو ستون برای نمایش هست که بخش دوم هست برای نمودار خط بعدی

# plt.plot(np.arange(len(city_name_y)), city_pop_y, ls='--' , marker='*', mew=5) # نمایش نمودار به صورت نقطه ای 

# plt.legend(['City X', 'City Y'], loc='best') # نمایش مشخصات نمودار

# plt.xticks(np.arange(len(city_name)), city_name)

# plt.grid()

# plt.yticks(city_pop, ['8m+', '8m+ < x < 5m', '5m < x < 3m', '3m < x < 1m'])

# plt.title('POP City Range...!')

# plt.text(0, 878515214, 'POP All City Soshal...!') # قرار دادن یک متن روی نمودار 

# plt.margins(0.3) # فاصله دادن از بغل های نمودار برای نمایش

# plt.show()

#---------------END Section 13, 14, 15, 16 ML------------------------


#---------------START Section 17, 18 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# #sb.boxplot(x='sepal_length', y='sepal_width', data=data) # تولد نمودار boxplot

# sb.pairplot(data, hue="variety") # تولید نمودار pairplot

# plt.show()

#---------------END Section 17, 18 ML------------------------


#---------------START Section 19, 20 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# counts = data.variety.value_counts() # نمایش تعداد و نوع ویژگی مورد نظر در ایجا گفته شده ابتدا از data بیا مقدار \variety رو بگیر بعد تعداد هر کدوم رو بده

# index = counts.index # نمایش مقدار های اصلی

# plt.bar(counts, index) # طراحی نمدار bar

# plt.show()

#---------------END Section 19, 20 ML------------------------


#---------------START Section 21 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# def ECDF(data):
#     n = len(data)
#     x = np.sort(data)
#     y = np.arange(1, n+1)/n
#     return x,y

# x,y = ECDF(data.sepal_length) # ساخت نمودار توزیع اطلاعات برای قدار های مختلف

# plt.scatter(x, y)

# plt.xlabel('sepal length')

# plt.ylabel('ECDF')

# plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5 ,0.6 , 0.7, 0.8, 0.9, 1], ['0%', '10%', '20%', '30%', '40%', '50%' ,'60%' , '70%', '80%', '90%', '100%'])

# plt.grid()

# plt.show()

#---------------END Section 21 ML------------------------


#---------------START Section 22 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# var = np.var(data.sepal_length) # واریانس مشخص میکنه چقدر داد های ما از میانگی داده ها فاصله دارند

# std = np.std(data.sepal_length) # انحراف از معیار که همان جذر واراینس است

# per = np.percentile(data.sepal_length, [25, 50, 75]) # اگر مقدار های 25و 50و 75 را بدهیم همان مقدار های Q1 ,Q2 , Q3 را میدهد که در boxplot نیاز بود در واقع Q1 کوچکتر ار 25 درصد از داده ها هست ً2 بزرکتر از 50 در است ً3 در واقع 75 درصد بزرگتر است

# print('var=>', var)

# print('std=>', std)

# print('per=>', per)

#---------------END Section 22 ML------------------------


#---------------START Section 23 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# plt.scatter(data.sepal_length, data.petal_length) # ساخت نمودار که قبلا ت.وضوح داده شده

# SL_mean = np.mean(data.sepal_length) # نقطه وسط ویژگی مورد نظر

# PL_mean = np.mean(data.petal_length) # همان بالای است

# print(SL_mean, '----', PL_mean)

# print(np.cov(data.sepal_length, data.petal_length)) # بدست اوردن کواریانس دوتا داده نسبت به هم 

# plt.plot(SL_mean, PL_mean, marker='o', color='red') # زدن نقطه بر اساس داده های داده شده

# plt.xlabel('sepal length')

# plt.ylabel('petal length')

# plt.grid()

# plt.show()

#---------------END Section 23 ML------------------------


#---------------START Section 24 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# plt.scatter(data.petal_length, data.petal_width) # ساخت نمودار که قبلا ت.وضوح داده شده

# p_c , p_v = pearsonr(data.petal_length, data.petal_width)

# print(p_c)

# # sb.pairplot(data)

# plt.show()

#---------------END Section 24 ML------------------------


#---------------START Section 25 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# data_2 = data.drop(['variety', 'sepal_length', 'sepal_width'], axis=1)

# corr = data_2.corr() # نمایش ارتباط بین ستون های مختلف باهم

# sb.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1)

# plt.show()

# print(corr)

#---------------END Section 25 ML------------------------


#---------------START Section 26, 27, 28, 29 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# t_b = pd.crosstab(data.petal_length, data.variety) # با استفاده از این تابع ما دوتا ویژگی را تعداد هر مقدار را برای هر دو ویژگی برسی می شود مثلا برای ویژگی 1 چه تعداد از ویژگی 2 وجود دارد

# chi , p_v , dof , t_e = chi2_contingency(t_b.values) # مقدار های متغییر بالا را میدهیم به این تابع و مقدار های که میخواهیم برای میزان شباهت دو متغییر را مشخص میکند (متغییر اول مقدار شباهت) (مقدار دوم مقدار جقدر شبیه هستند )( مقدار سوم چقدر درسته فرضیه صفر)(و مقدار اخر جدول درستی را نشان میدهد)

# print(t_e)

#---------------END Section 26, 27, 28, 29 ML------------------------


#---------------START Section 30, 31 ML------------------------

# ss = np.random.normal(0, 1, size=1000) # توزیع نرمال عدد ها به صورت تصادفی

# sb.distplot(ss) # ساخت نمودار برای توزیع نرمال

# plt.show()

#---------------END Section 30, 31 ML------------------------


#---------------START Section 32, 33, 34 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# data.replace(1.4, '?', inplace=True) # تغییر داده ها به ؟ برای این که داده های missing بسازیم و در ادامه نرمال کنیم

# data.drop(['sepal_length', 'sepal_width'], axis=1, inplace=True) # یکی از عملیات های پیش پردازش داده است برای کاهش ابعداد داده به ستون یا ویژگی های که نیاز نداریم اون ها رو پاک میکنیم

# # یکی دیگر از عملیات حذف داده های نادرست است که در row ها هست در این دیتا ست دیتا ها همه دست است و نیازی به حذف نیاز (داده نادرست یعننی مثلا جمعیت یک شهر مساوی با جمعیت کل دنیا باشد!!!!)

# data.replace('?', np.nan, inplace=True) # برای درست کردن داده های خطا دارد یا گم شده ابتدا ان ها را با مقدار خالی که numpy است پر میکنیم 

# data.petal_length.fillna(data.petal_length.mean(), inplace=True) # قرار دادن میانگین داده های petal_length جای مقدار های NaN

# data.petal_width.fillna(data.petal_width.mean(), inplace=True) # قرار دادن میانگین داده های petal_width جای مقدار های NaN

# # اگر بخواهیم این کار را برای هر ستون تکرار نکینم باید برای کل دیتافریم این کار را انجام دهیم پس جا است به این شکل کار میکنیم data.fillna(data.mean(), inplace=True)

# print(data.isnull().sum())

#---------------END Section 32, 33, 34 ML------------------------


#---------------START Section 32, 33, 34, 35, 36 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# data.replace(1.4, '?', inplace=True) # تغییر داده ها به ؟ برای این که داده های missing بسازیم و در ادامه نرمال کنیم

# data.drop(['sepal_length', 'sepal_width'], axis=1, inplace=True) # یکی از عملیات های پیش پردازش داده است برای کاهش ابعداد داده به ستون یا ویژگی های که نیاز نداریم اون ها رو پاک میکنیم

# # یکی دیگر از عملیات حذف داده های نادرست است که در row ها هست در این دیتا ست دیتا ها همه دست است و نیازی به حذف نیاز (داده نادرست یعننی مثلا جمعیت یک شهر مساوی با جمعیت کل دنیا باشد!!!!)

# data.replace('?', np.nan, inplace=True) # برای درست کردن داده های خطا دارد یا گم شده ابتدا ان ها را با مقدار خالی که numpy است پر میکنیم 

# # data.fillna({'petal_length':1.111, 'petal_width':1.111}, inplace=True) # قرار دادن مقدار های متفاوت برای هر ستون خاص

# # i = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0) # قرار دادن مقدار میانگین برای مقدار های خالی

# # i.fit(data) # فیت کردن مشخصات قبلی روی دیتافریم

# # new_data = i.transform(data) # ذخیره تغییرات جدید در یک متغییر

# # d_data = data.drop_duplicates() # حذف داده های تکراری در تماما ستون ها

# # d_data_2 = data.drop_duplicates(['petal_length']) # حذف داده های تکراری فقط برای ستون petal_length

# # new_data = pd.concat([data, data_2], axis=0, ignore_index=True) # ترکیب دو دیتافریم باهم مقدار axis میگه به سطر ها اضافه شود وl مقدارignore_index میگه که از اول شماره index  ها رو بساز (بعد از این کار باید ستونی که خیلی )

# print(data)

#---------------END Section 32, 33, 34, 35, 36 ML------------------------


#---------------START Section 37 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# val = data.variety.value_counts() #  گرفتن اطلعات مختصر برای ویژگی خاص

# gro = data.groupby(data.variety) # مقدار وردی مارو با مقدار های کلی دیتافریم برسی میکنه مثلا با استفاده از تابع mean میتوان گقت که هر یک از مقدار های ورودی را برای هر یک از مقدار های دیتافریم میانگین میگیرد(مثلا میگه در سیستم عامل ویندوز که ورودی هست در رم 8 گیگ میانگین انقدر هست)

# print(gro.mean())

#---------------END Section 37 ML------------------------


#---------------START Section 38 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# counter = pd.crosstab(data.variety, data.petal_width) # این متدد میشماره از ویژگی اولی در ویژگی دومی چندتا هست

# counter_pivote = pd.pivot_table(data, index=data.variety , columns=data.petal_width, values=data.petal_length) # برسی میکنه برای ویژگی ایندکس و در کول چه تعداد ولیو هست

# print(counter_pivote)

#---------------END Section 38 ML------------------------



#---------------START Section 39 ML------------------------

# data = pd.read_csv('iris.csv')

# data.rename(columns={'sepal.length' : 'sepal_length', 'sepal.width' : 'sepal_width', 'petal.length' : 'petal_length', 'petal.width' : 'petal_width'}, inplace=True)

# new_data = pd.get_dummies(data) # تبدیل کردن ویژگی های که عدد نیستند و خیلی راحت تامام وی"ی های که عدد نیستن را به چند ویژگی تبدیل میکنه و هر جای که اون ویژگی را داشت میاد یک جلوش میزاره

# print(new_data)

#---------------END Section 39 ML------------------------