import pyod.utils.data as pyod
import matplotlib.pyplot as plt
X_train, X_test, Y_train, Y_test=pyod.generate_data(n_train=400, n_test=100,contamination=0.1)
for i in range(len(Y_train)):
    if Y_train[i]==1:
        plt.scatter(X_train[i,0],X_train[i,1],color='blue')
    else:
        plt.scatter(X_train[i,0],X_train[i,1],color='green')
plt.show()