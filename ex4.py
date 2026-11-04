import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
from pyod.models.combination import average,maximization
cardio_dataset=sio.loadmat("cardio.mat")
X=cardio_dataset['X']
print(X.shape)
Y=cardio_dataset['y']
print(Y.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.60,shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
X_train_norm,X_test_norm=standardizer(X_train,X_test)
neighbors=np.linspace(30,120,11)
train_knn_scores=[]
train_lof_scores=[]
test_knn_scores=[]
test_lof_scores=[]
for k in neighbors:
  k=int(k)
  print(k)
  knn=KNN(n_neighbors=k)
  lof=LOF(n_neighbors=k)
  knn.fit(X_train_norm)
  lof.fit(X_train_norm)
  Y_pred_train_knn=knn.predict(X_train_norm)
  Y_pred_test_knn=knn.predict(X_test_norm)
  Y_pred_train_lof=lof.predict(X_train_norm)
  Y_pred_test_lof=lof.predict(X_test_norm)
  ba_train_knn=balanced_accuracy_score(Y_train,Y_pred_train_knn)
  print(ba_train_knn)
  train_knn_scores.append(knn.decision_scores_)
  ba_train_lof=balanced_accuracy_score(Y_train,Y_pred_train_lof)
  print(ba_train_lof)
  train_lof_scores.append(lof.decision_scores_)
  ba_test_knn=balanced_accuracy_score(Y_test,Y_pred_test_knn)
  print(ba_test_knn)
  test_knn_scores.append(knn.decision_function(X_test_norm))
  ba_test_lof=balanced_accuracy_score(Y_test,Y_pred_test_lof)
  print(ba_test_lof)
  test_lof_scores.append(lof.decision_function(X_test_norm))
train_knn_scores_matrix=np.column_stack(train_knn_scores)
test_knn_scores_matrix=np.column_stack(test_knn_scores)
knn_scores_train_norm,knn_scores_test_norm=standardizer(train_knn_scores_matrix,test_knn_scores_matrix)
train_lof_scores_matrix=np.column_stack(train_lof_scores)
test_lof_scores_matrix=np.column_stack(test_lof_scores)
lof_scores_train_norm,lof_scores_test_norm=standardizer(train_lof_scores_matrix,test_lof_scores_matrix)
knn_scores_avg_train=average(knn_scores_train_norm)
knn_scores_avg_test=average(knn_scores_test_norm)
lof_scores_avg_train=average(lof_scores_train_norm)
lof_scores_avg_test=average(lof_scores_test_norm)
knn_scores_max_train=maximization(knn_scores_train_norm)
knn_scores_max_test=maximization(knn_scores_test_norm)
lof_scores_max_train=maximization(lof_scores_train_norm)
lof_scores_max_test=maximization(lof_scores_test_norm)
contamination_rate=0.096
thr_knn_max=np.quantile(knn_scores_max_train,1-contamination_rate)
print(thr_knn_max)
thr_lof_max=np.quantile(lof_scores_max_train,1-contamination_rate)
print(thr_lof_max)
thr_knn_avg=np.quantile(knn_scores_avg_train,1-contamination_rate)
print(thr_knn_avg)
thr_lof_avg=np.quantile(lof_scores_avg_train,1-contamination_rate)
print(thr_lof_avg)
pred_knn_max_train=[1 if score>=thr_knn_max else 0 for score in knn_scores_max_train]
ba_knn_max_train=balanced_accuracy_score(Y_train,pred_knn_max_train)
print(ba_knn_max_train)
pred_knn_max_test=[1 if score>=thr_knn_max else 0 for score in knn_scores_max_test]
ba_knn_max_test=balanced_accuracy_score(Y_test,pred_knn_max_test)
print(ba_knn_max_test)
pred_lof_max_train=[1 if score>=thr_lof_max else 0 for score in lof_scores_max_train]
ba_lof_max_train=balanced_accuracy_score(Y_train,pred_lof_max_train)
print(ba_lof_max_train)
pred_lof_max_test=[1 if score>=thr_lof_max else 0 for score in lof_scores_max_test]
ba_lof_max_test=balanced_accuracy_score(Y_test,pred_lof_max_test)
print(ba_lof_max_test)
pred_knn_avg_train=[1 if score>=thr_knn_avg else 0 for score in knn_scores_avg_train]
ba_knn_avg_train=balanced_accuracy_score(Y_train,pred_knn_avg_train)
print(ba_knn_avg_train)
pred_knn_avg_test=[1 if score>=thr_knn_avg else 0 for score in knn_scores_avg_test]
ba_knn_avg_test=balanced_accuracy_score(Y_test,pred_knn_avg_test)
print(ba_knn_avg_test)
pred_lof_avg_train=[1 if score>=thr_lof_avg else 0 for score in lof_scores_avg_train]
ba_lof_avg_train=balanced_accuracy_score(Y_train,pred_lof_avg_train)
print(ba_lof_avg_train)
pred_lof_avg_test=[1 if score>=thr_lof_avg else 0 for score in lof_scores_avg_test]
ba_lof_avg_test=balanced_accuracy_score(Y_test,pred_lof_avg_test)
print(ba_lof_avg_test)