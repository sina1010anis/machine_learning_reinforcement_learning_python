import numpy as np
import matplotlib.pyplot as plt
import ProcessesKNN as KNN
import ProcessesLR as LR
import ProcessesNB as NB

S_KNN = KNN.showPlot()
S_LR = LR.showPlot()
S_NB = NB.showPlot()
n = np.arange(len(S_KNN))

plt.title('CatchPhish D1')

plt.plot(n, S_KNN, marker='o', linewidth=3.0)

plt.plot(n, S_LR, marker='o', linewidth=3.0)

plt.plot(n, S_NB, marker='o', linewidth=3.0)

plt.xticks(n, ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'F1_score', 'ROC', 'Accuracy', 'Score'])

plt.yticks([0, 10, 20 ,30 ,40,50,60,70,80,90,100], ['0%', '10%', '20%' ,'30%' ,'40%','50%','60%','70%','80%','90%','100%'])

plt.legend(['KNeighbors', 'Logistic Regression', 'Naive Bayes'], loc='best')

plt.grid()

plt.margins(0.1)

plt.show()



