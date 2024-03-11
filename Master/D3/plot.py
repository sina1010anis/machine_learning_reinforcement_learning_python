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

plt.title('CatchPhish D3 (My Score)')

plt.plot(n, S_KNN, marker='o', linewidth=2)

plt.plot(n, S_LR, marker='o', linewidth=2)

plt.plot(n, S_NB, marker='o', linewidth=2)

plt.plot(n, S_RF, marker='o', linewidth=2)

plt.plot(n, S_ADA, marker='o', linewidth=2)

plt.plot(n, [93.88,25.85,74.15,6.12,88.51,91.11,81.2,87.5,87.5], marker='o', linewidth=2)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.yticks([0, 10, 20 ,30 ,40,50,60,70,80,90,100], ['0%', '10%', '20%' ,'30%' ,'40%','50%','60%','70%','80%','90%','100%'])

plt.legend(['KNeighbors', 'Logistic Regression', 'Naive Bayes','Random Forest','AdaBoosting' , 'SVM'], loc='best')

plt.grid()

plt.margins(0.1)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.subplot(1, 2, 2)

plt.title('CatchPhish D3 (Arc Score)')

plt.plot(n, [81.1,8.4,91.6,18.9,82.1,81.6,87.5,88.9,88.9], marker='o', linewidth=2)

plt.plot(n, [66,6,94.0,34,83.9,73.9,90.7,85.01,85.01], marker='o', linewidth=2)

plt.plot(n, [43.5,6.2,93.2,66.5,76.9,55.6,84.5,77.6,77.6], marker='o', linewidth=2)

plt.plot(n, [86.1,4.9,95.1,19.9,89.3,87.5,96.3,92.2,92.2], marker='o', linewidth=2)

plt.plot(n, [47.2,3.6,96.4,13.9,86.2,61,87.2,80.61,80.61], marker='o', linewidth=2)

plt.plot(n, [53.7,3.4,96.6,46.3,88.3,66.8,75.2,82.84,82.84], marker='o', linewidth=2)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.yticks([0, 10, 20 ,30 ,40,50,60,70,80,90,100], ['0%', '10%', '20%' ,'30%' ,'40%','50%','60%','70%','80%','90%','100%'])

plt.legend(['KNeighbors', 'Logistic Regression', 'Naive Bayes','Random Forest','AdaBoosting', 'SVM'], loc='best')

plt.grid()

plt.margins(0.1)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.show()



