import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def buildData(address_file, data_drop_items=[]):

    data = pd.read_csv(address_file)

    data.rename(index=data.URL, inplace=True)

    labels = np.array(data.label)

    data_drop = ['label', 'URL']

    if len(data_drop_items) > 0:

        for i in data_drop_items:

            data_drop.append(i)

    data.drop(data_drop, axis=1, inplace=True)

    data_x = np.array(data)

    x_tr, x_te, l_tr, l_te = train_test_split(data, labels, test_size=0.1, train_size=0.9, random_state=42) # Train = 90% & Test = 10%
    # x_tr, x_te, l_tr, l_te = train_test_split(data, labels, test_size=0.3, train_size=0.7, random_state=42) # Train = 70% & Test = 30%

    # x_tr, x_te, l_tr, l_te = train_test_split(data, labels, test_size=0.3, shuffle=True)

    return x_tr, x_te, l_tr, l_te, data_x, labels, data

def score(method, x_te, l_te, l_pre, mode='print'):

    cnf_matrix = confusion_matrix(l_te, l_pre)

    cr = classification_report(l_te, l_pre)

    # l_pre_pro = method.predict_proba(x_te)[:,1]

    # roc = roc_auc_score(l_te, l_pre_pro)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    F1 = (2*TP)/((2*TP)+FP+FN)
    TPR = (TP/(TP+FN))*100
    FPR =(FP/(FP+TN))*100
    TNR =(TN/(TN+FP))*100
    FNR =(FN/(TP+FN))*100
    Precision =(TP/(TP+FP))*100
    ACC =((TP+TN)/(TP+FP+FN+TN))*100
    score = method.score(x_te, l_te)*100
    if mode == 'print':

        print("TPR", '(',round(TPR[0], 2),')')

        print("FPR", '(',round(FPR[0], 2),')')

        print("TNR", '(',round(TNR[0], 2),')')

        print("FNR", '(',round(FNR[0], 2),')')

        print("Precision", '(',round(Precision[0], 2),')')

        print("F1-score", '(',round(F1[0]*100, 2),')')

        # print("ROC", '(',round(roc*100, 2),')')

        print("Accuracy", '(',round(ACC[0], 2),')')

        print("Score", '(',round(score, 2),')')

    elif mode == 'return':

        return [round(TPR[0], 2), round(FPR[0], 2), round(TNR[0], 2), round(FNR[0],2 ), round(Precision[0], 2), round(F1[0]*100, 2), round(roc*100, 2),round(ACC[0], 2),round(score, 2)]

