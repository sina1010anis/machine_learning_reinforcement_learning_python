import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('phishing.csv')

data.replace({'legitimate':0, 'phishing':1}, inplace=True)

data.drop_duplicates(['url'])

data.rename(index=data.url, inplace=True)

labels = np.array(data.status)

data.drop(['length_url','url' , 'status', 'ip', 'nb_tilde', 'nb_space', 'nb_www', 'ratio_digits_host','ratio_digits_url', 'punycode', 'port', 'tld_in_subdomain', 'nb_subdomains', 'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension', 'length_words_raw', 'shortest_word_host', 'shortest_words_raw', 'shortest_word_path', 'longest_words_raw', 'longest_word_path', 'avg_words_raw', 'avg_word_path', 'domain_in_brand', 'statistical_report', 'nb_hyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS', 'ratio_extRedirection', 'ratio_extErrors', 'login_form', 'external_favicon', 'links_in_tags', 'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 'safe_anchor', 'right_clic', 'empty_title', 'domain_in_title', 'domain_registration_length', 'dns_record', 'google_index'], axis=1, inplace=True) # حذف ویژگی های که نیاز نیست برای کاهش ابعاد

data.drop_duplicates()

data.replace('?', np.nan)

data.dropna()

data_x = np.array(data)

x = minmax_scale(data_x, feature_range=(0, 1))

x_tr, x_te, l_tr, l_te = train_test_split(x, labels, test_size=0.3, shuffle=True) 

############################ Lasso ############################

# lasso = Lasso(alpha=0.1)

# lasso.fit(x_tr, l_tr)

# print('Lasso => %', int(lasso.score(x_te, l_te)*100))

############################ Ridge ############################

ridge = Ridge(alpha=0.1)

ridge.fit(x_tr, l_tr)

key_arr = []

val_arr = []

tr_score = []

te_score = []

j = 0

t = 0

for i in ridge.coef_:

    if i < 0.0001 :

        key_arr.append(j)
        
        t = t + 1

    j = j + 1


for s in key_arr:

    val_arr.append(str(data.columns[s]))

data.drop(val_arr, axis=1, inplace=True)

x_new = minmax_scale(data, feature_range=(0, 1))

x_tr, x_te, l_tr, l_te = train_test_split(x_new, labels, test_size=0.3, shuffle=True) 

n = np.arange(1, 31)

for i in range(30):
    
    x_tr, x_te, l_tr, l_te = train_test_split(x_new, labels, test_size=0.3, shuffle=True) 

    knn = KNeighborsClassifier(n_neighbors=i+1) # کانفیگ روش knn

    knn.fit(x_tr, l_tr) # فیت کردن داده و برچسب ها
    
    tr_score.append(knn.score(x_tr, l_tr))
    
    te_score.append(knn.score(x_te, l_te))

plt.plot(n, tr_score, label="Train Data")

plt.plot(n, te_score, label="Test Data")

plt.legend()

plt.xlabel('Number of K (Ridge)')

plt.ylabel('Progress')

plt.title('Knn Best K')

plt.grid()

plt.show()

