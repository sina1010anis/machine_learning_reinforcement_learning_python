import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,scale,StandardScaler

db_csv = pd.read_csv('iris.csv')

db_excel = pd.read_excel('Fruit_data.xlsx')

db_excel = db_excel.replace('?', np.nan) # جای گذاری مقدار nan به جای ؟

db_excel = db_excel.fillna({'mass' : np.mean(db_excel['mass']), 'width' : np.mean(db_excel['width']), 'height' : np.mean(db_excel['height']), 'color_score' : np.mean(db_excel['color_score'])}) # داده های که در مرحله قبلی nan گذاشتیم را حال تغییر میدهیم به مقدار میانگین همان ستون تا داده ها را از دست ندهیم

db_excel['fruit_name'] = (LabelEncoder()).fit_transform(db_excel['fruit_name']) # تغیر رنج متن ها به عدد

db_excel['fruit_subtype'] = (LabelEncoder()).fit_transform(db_excel['fruit_subtype']) # تغیر رنج متن ها به عدد

db_excel = np.log(db_excel)

a = StandardScaler().fit(db_excel).transform(db_excel)

db_excel = pd.DataFrame(a, columns=db_excel.columns)

print(type(db_excel))