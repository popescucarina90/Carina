from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
X,Y=make_blobs(n_samples=[200,100],n_features=2,center_box=(-10,10),cluster_std=[2,6])
n_neighbors=[1,3,5,7,9,11,13,15,17,19,21]
for k in n_neighbors:
    knn=KNN(contamination=0.07,n_neighbors=k)
    lof=LOF(contamination=0.07,n_neighbors=k)
    knn.fit(X)
    lof.fit(X)
    y_pred_knn=knn.predict(X)
    y_pred_lof=lof.predict(X)
    fig,ax=plt.subplots(1,2,figsize=(12,6))
    ax[0].scatter(X[:,0],X[:,1],c=y_pred_knn)
    ax[0].set_title(f'KNN cu k={k}')
    ax[1].scatter(X[:,0],X[:,1],c=y_pred_lof)
    ax[1].set_title(f'LOF cu k={k}')
    plt.show()