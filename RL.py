print('\n')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxscaler


dataset = pd.read_excel('./Fruit_data.xlsx')  # خواندن فایل اکسل

dataset = dataset.replace('?', np.nan) # جای گذاری داداه های که با ؟ مقدار دهی شده انند با nan در numpy

dataset = dataset.dropna() # حذف داده های که nan هستند

dataset = dataset.set_index(np.arange(0, len(dataset))) # چون داده ها حذف شده انند حتما ایدی یا ایندس ان ها ئجار اشتباه شده پس مجدد میگیم این مقدار های مف=قدار دهی شود میگیم ابتدا از 0 شروع کند تا تعداد خود دیتاست شماره گذاری کنه

dataset = dataset.drop('fruit_subtype', axis = 1) # حذف کردن یک ستونی که به هدف ما ربطی تنداره

dataset = dataset.reindex(['fruit_name', 'mass', 'width', 'height', 'color_score', 'fruit_label'], axis=1) # تغییر ترتیب سوتون ها (axis چون یک شده است که قرارا روی ستون ها کار کنه)

## ---------------- شروع--------------------------- کار تبدیل متن داده به عدد و اجرا رنج بین 0 و یک و تبدیل مجدد به دیتا فریم

dataset['fruit_name'] = LabelEncoder.fit_transform(dataset['fruit_name']) # تغیر رنج متن ها به عدد

obj = MinMaxscaler(featrue_rang=(0, 1)) # تغییر رنج عدد ها بین یک بازه خاص مصلا 0 و 0

obj = obj.fit(dataset) # اجرای بازه ساخته شده روی دیتا موردنظر

normal = obj.transform(dataset) # ترنسفور کردن داده

dataset = pd.DataFrame(normal, columns=['fruit_name', 'mass', 'width', 'height', 'color_score', 'fruit_label']) # تبدیل مجدد داده به دیتا فریم چون این داده تبدیل به عدد شده دیگه از حالت دیتافریم خارج شده 

## ---------------- اتمام----------------------------- کار تبدیل متن داده به عدد و اجرا رنج بین 0 و یک و تبدیل مجدد به دیتا فریم