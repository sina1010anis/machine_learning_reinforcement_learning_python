import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix , classification_report
import seaborn as sn

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Embedding, GlobalMaxPooling1D

df = pd.read_csv("drive/MyDrive/Thesis/3catchphish.csv")
#df = df.sample(n=40000)
#df.sample(5)

X = df.drop('class',axis='columns')
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

def createModel():
    model = Sequential()
    #model.add(Dense(1, input_shape=(100,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros'))
    model.add(Dense(100, input_shape=(100,), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

model = createModel()

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model.summary()
#keras.utils.plot_model(model1, "my_first_model_with_shape_info.png", show_shapes=True)
hist = model.fit(X_train, y_train, epochs=50)

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
#print(y_pred[:10])

# round the values to nearest integer ie 0 or 1
y_pred = np.round(y_pred)
#print(y_pred[:10])

print(classification_report(y_test, y_pred))


cnf_matrix = confusion_matrix(y_test,y_pred)

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("TPR", TPR)
# fpr
FPR = FP/(FP+TN)
print("FPR", FPR)
# Specificity or true negative rate
TNR = TN/(TN+FP)
print("TNR", TNR)
# False negative rate
FNR = FN/(TP+FN)
print("FNR", FNR)
# Precision or positive predictive value
Precision = TP/(TP+FP)
print("Precision", Precision)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Accuracy", ACC)