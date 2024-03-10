import numpy as np
import matplotlib.pyplot as plt
import ProcessesKNN as KNN
import ProcessesLR as LR
import ProcessesNB as NB
import ProcessesRF as RF
import ProcessesADA as ADA

S_KNN = KNN.showPlot()
S_LR = LR.showPlot()
S_NB = NB.showPlot()
S_RF = RF.showPlot()
S_ADA = ADA.showPlot()
n = np.arange(len(S_KNN))



plt.subplot(1, 2, 1)

plt.title('CatchPhish D2 (My Score)')

plt.plot(n, S_KNN, marker='o', linewidth=2)

plt.plot(n, S_LR, marker='x', linewidth=2)

plt.plot(n, S_NB, marker='v', linewidth=2)

plt.plot(n, S_RF, marker='^', linewidth=2)

plt.plot(n, S_ADA, marker='*', linewidth=2)

plt.plot(n, [87.89,12.06,87.94,12.11,88.36,88.12,87.92,87.92,87.92], marker='+', linewidth=2)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.yticks([0, 10, 20 ,30 ,40,50,60,70,80,90,100], ['0%', '10%', '20%' ,'30%' ,'40%','50%','60%','70%','80%','90%','100%'])

plt.legend(['KNeighbors', 'Logistic Regression', 'Naive Bayes','Random Forest','AdaBoosting' , 'SVM'], loc='best')

plt.grid()

plt.margins(0.1)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.subplot(1, 2, 2)

plt.title('CatchPhish D1 (Arc Score)')

plt.plot(n, [86.7,14.2,85.8,13.3,85.8,86.3,86.9,86.28,86.28], marker='o', linewidth=2)

plt.plot(n, [82.6,13.3,87.7,17.4,86,84.2,92.4,84.65,84.65], marker='x', linewidth=2)

plt.plot(n, [50,8.5,91.5,50,85.3,63,87.5,70.91,70.91], marker='v', linewidth=2)

plt.plot(n, [91.5,8.2,91.8,8.5,91.7,91.5,97.2,91.63,91.63], marker='^', linewidth=2)

plt.plot(n, [78.1,11.4,88.6,21.9,87.1,82.3,83.3,83.31,83.31], marker='*', linewidth=2)

plt.plot(n, [80.1,20.9,79.1,19.9,46.1,79.6,87.6,72.62,72.62], marker='+', linewidth=2)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.yticks([0, 10, 20 ,30 ,40,50,60,70,80,90,100], ['0%', '10%', '20%' ,'30%' ,'40%','50%','60%','70%','80%','90%','100%'])

plt.legend(['KNeighbors', 'Logistic Regression', 'Naive Bayes','Random Forest','AdaBoosting', 'SVM'], loc='best')

plt.grid()

plt.margins(0.1)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.show()



