import pyod.utils.data as data
from pyod.models.knn import KNN
import matplotlib.pyplot as plt 
X_train,X_test,Y_train,Y_test=data.generate_data_clusters(n_features=2,n_train=400,n_test=200,n_clusters=2,contamination=0.1)
knn=KNN()
knn.fit(X_train)
Y_pred_test=knn.predict(X_test)
Y_pred_train=knn.predict(X_train)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,9))
#1
plt.subplot(2,2,1)
plt.scatter(X_train[:,0],X_train[:,1],c=Y_train)
plt.title("Train-Ground Truth")
#2
plt.subplot(2,2,2)
plt.scatter(X_train[:,0],X_train[:,1],c=Y_pred_train)
plt.title("Train-Predicted Truth")
#3
plt.subplot(2,2,3)
plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
plt.title("Test-Ground Truth")
#4
plt.subplot(2,2,4)
plt.scatter(X_test[:,0],X_test[:,1],c=Y_pred_test)
plt.title("Test-Predicted Truth")
plt.show()
n=[1,3,5,7,9,13,15,17,19,21]
for k in n:
    print("K=",k)
    knn=KNN(n_neighbors=k)
    knn.fit(X_train)
    Y_pred_test=knn.predict(X_test)
    Y_pred_train=knn.predict(X_train)
    TN_train=0
    TP_train=0
    FN_train=0
    FP_train=0
    for i in range(len(Y_train)):
        if Y_train[i]==1 and Y_pred_train[i]==1:
            TP_train+=1
        elif Y_train[i]==1 and Y_pred_train[i]==0:
            FN_train+=1
        elif Y_train[i]==0 and Y_pred_train[i]==1:
            FP_train+=1
        elif Y_train[i]==0 and Y_pred_train[i]==0:
            TN_train+=1
    TPR_train=TP_train/(TP_train+FN_train)
    TNR_train=TN_train/(TN_train+FP_train)
    BA_train=(TNR_train+TPR_train)/2
    print("Train")
    print("K=",k," BA=",BA_train)
    TN_test=0
    TP_test=0
    FN_test=0
    FP_test=0
    for i in range(len(Y_test)):
        if Y_test[i]==1 and Y_pred_test[i]==1:
            TP_test+=1
        elif Y_test[i]==1 and Y_pred_test[i]==0:
            FN_test+=1
        elif Y_test[i]==0 and Y_pred_test[i]==1:
            FP_test+=1
        elif Y_test[i]==0 and Y_pred_test[i]==0:
            TN_test+=1
    TPR_test=TP_test/(TP_test+FN_test)
    TNR_test=TN_test/(TN_test+FP_test)
    BA_test=(TNR_test+TPR_test)/2
    print("Test")
    print("K=",k," BA=",BA_test)
    plt.figure(figsize=(12,9))
    #1
    plt.subplot(2,2,1)
    plt.scatter(X_train[:,0],X_train[:,1],c=Y_train)
    plt.title("Train-Ground Truth")
    #2
    plt.subplot(2,2,2)
    plt.scatter(X_train[:,0],X_train[:,1],c=Y_pred_train)
    plt.title("Train-Predicted Truth")
    #3
    plt.subplot(2,2,3)
    plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
    plt.title("Test-Ground Truth")
    #4
    plt.subplot(2,2,4)
    plt.scatter(X_test[:,0],X_test[:,1],c=Y_pred_test)
    plt.title("Test-Predicted Truth")
    plt.suptitle(f"KNN-n_neighbors={k}")
    plt.show()