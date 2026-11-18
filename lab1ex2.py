import pyod.utils.data as pyod
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix,roc_curve
import matplotlib.pyplot as plt
X_train,X_test,Y_train,Y_test=pyod.generate_data(n_train=400,n_test=100,contamination=0.1)
knn=KNN(contamination=0.1)
knn.fit(X_train)
y_guess_train=knn.predict(X_train)
y_guess_test=knn.predict(X_test)
confusion=confusion_matrix(Y_test,y_guess_test)
print(confusion)
tp=confusion[0,0]
tn=confusion[1,1]
fp=confusion[1,0]
fn=confusion[0,1]
tpr=tp/(tp+fn)
tnr=tn/(tn+fp)
ba=(tpr+tnr)/2
fpr,tpr,thresholds=roc_curve(Y_train,knn.decision_scores_)
plt.plot(fpr,tpr)
plt.show()
