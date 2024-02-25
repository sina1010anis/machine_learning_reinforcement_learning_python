import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import pearsonr,chi2_contingency
from sklearn import preprocessing, datasets
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('phishing.csv')

data.replace({'legitimate':0, 'phishing':1}, inplace=True) # تغییر برچسب ها به عدد

data.drop_duplicates(['url']) # حدف ادرس های تکراری کاهش حجم دیتاست 

data.rename(index=data.url, inplace=True) # تغییر نام ایندکس به جای ادرس

data_label = np.array(data.status) # ریختن برچسب ها در متغییر به دلیل استفاده در تشخیص 

data.drop(['length_url','url' , 'ip', 'nb_tilde', 'nb_space', 'nb_www', 'ratio_digits_host','ratio_digits_url', 'punycode', 'port', 'tld_in_subdomain', 'nb_subdomains', 'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension', 'length_words_raw', 'shortest_word_host', 'shortest_words_raw', 'shortest_word_path', 'longest_words_raw', 'longest_word_path', 'avg_words_raw', 'avg_word_path', 'domain_in_brand', 'statistical_report', 'nb_hyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS', 'ratio_extRedirection', 'ratio_extErrors', 'login_form', 'external_favicon', 'links_in_tags', 'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 'safe_anchor', 'right_clic', 'empty_title', 'domain_in_title', 'domain_registration_length', 'dns_record', 'google_index'], axis=1, inplace=True) # حذف ویژگی های که نیاز نیست برای کاهش ابعاد

data.drop_duplicates() # حذف داده های تکرای کم کردن داده ها

data.replace('?', np.nan) # قرار دادن مقدار خالی برای داده های گم شده

data.dropna() # حذف داده های خالی که در قبل گفته شد

data_x = np.array(data) # قرار دادن داده ها داخل یک متغییر برای تشخیص

data_normal = minmax_scale(data_x, feature_range=(0, 1)) # قرار دادن داده ها بین عدد صفر و یک برای نرمال سازی


#-------------------- KNN ---------------------------

tr_score = np.empty(30)

te_score = np.empty(30)

n = np.arange(1, 31)

for i in range(30):
    
    x_tr, x_te, y_tr, y_te = train_test_split(data_normal, data_label, test_size=0.3, random_state=42, stratify=data.status) # تقسیم داده ها برای تست و اموزش 

    knn = KNeighborsClassifier(n_neighbors=i+1) # کانفیگ روش knn

    knn.fit(x_tr, y_tr) # فیت کردن داده و برچسب ها
    
    tr_score[i] = knn.score(x_tr, y_tr)
    
    te_score[i] = knn.score(x_te, y_te)

plt.plot(n, tr_score, label="Train Data")

plt.plot(n, te_score, label="Test Data")

plt.legend()

plt.xlabel('Number of K')

plt.ylabel('Progress')

plt.title('Knn Best K')

plt.grid()

plt.show()