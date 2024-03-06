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

plt.title('CatchPhish D1 (My Score)')

plt.plot(n, S_KNN, marker='o', linewidth=2)

plt.plot(n, S_LR, marker='o', linewidth=2)

plt.plot(n, S_NB, marker='o', linewidth=2)

plt.plot(n, S_RF, marker='o', linewidth=2)

plt.plot(n, S_ADA, marker='o', linewidth=2)

plt.plot(n, [93.88, 5.09, 94.3, 6.16, 94.59, 94.84, 97.2, 94.7,94.7], marker='o', linewidth=2)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.yticks([0, 10, 20 ,30 ,40,50,60,70,80,90,100], ['0%', '10%', '20%' ,'30%' ,'40%','50%','60%','70%','80%','90%','100%'])

plt.legend(['KNeighbors', 'Logistic Regression', 'Naive Bayes','Random Forest','AdaBoosting' , 'SVM'], loc='best')

plt.grid()

plt.margins(0.1)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.subplot(1, 2, 2)

plt.title('CatchPhish D1 (Arc Score)')

plt.plot(n, [91.8,5.5, 94.5, 8.2,94.1,92.9,93.9,93.19,93.19], marker='o', linewidth=2)

plt.plot(n, [90, 5.1,94.9,9.1,94.4,92.6,97,92.93,92.93], marker='o', linewidth=2)

plt.plot(n, [72.7,3.4, 96.6, 27.3,95.4,72.7,95,84.95,84.95], marker='o', linewidth=2)

plt.plot(n, [94.6 ,5, 95, 5.4, 94.7, 94.7, 97.9 ,94.8, 94.8], marker='o', linewidth=2)

plt.plot(n, [88.7,5.5,94.5,11.3,93.9,91.2,96.5,91.68,91.68], marker='o', linewidth=2)

plt.plot(n, [91.5, 5.7, 94.3, 8.5,93.9,92.7,92.9,92.96,92.96], marker='o', linewidth=2)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.yticks([0, 10, 20 ,30 ,40,50,60,70,80,90,100], ['0%', '10%', '20%' ,'30%' ,'40%','50%','60%','70%','80%','90%','100%'])

plt.legend(['KNeighbors', 'Logistic Regression', 'Naive Bayes','Random Forest','AdaBoosting', 'SVM'], loc='best')

plt.grid()

plt.margins(0.1)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.show()



