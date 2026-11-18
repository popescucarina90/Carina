import pyod.utils.data as pyod
import numpy as np
X_train,X_test,Y_train,Y_test=pyod.generate_data(n_train=1000, n_test=0,contamination=0.1,n_features=1)
med=np.mean(X_train)
std=np.mean(X_train)
z_scores=np.abs(X_train - med)/std
z_scores=z_scores.reshape(-1)
threshold=np.quantile(z_scores,0.9)
tp=0
fn=0
fp=0
tn=0
for i in range(len(Y_train)):
    if Y_train[i]==1:
        if z_scores[i]>=threshold:
            tp+=1
        else:
            fn+=1
    else:
        if z_scores[1]>=threshold:
            fp+=1
        else:
            tn+=1
tpr=tp/(tp+fn)
tnr=tn/(tn+fp)
ba=(tpr+tnr)/2
print(f'Balanced accuracy:{ba}')