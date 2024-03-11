import ProcessesKNN as Knn
import ProcessesADA as Ada
import ProcessesLR as Lr
import ProcessesNB as Nb
import ProcessesRF as Rf

print('KNN =>')
print( Knn.benchmark())
print('ADA =>')
print( Ada.benchmark())
print('LR =>')
print( Lr.benchmark())
print('NB =>')
print( Nb.benchmark())
print('RF =>')
print( Rf.benchmark())
