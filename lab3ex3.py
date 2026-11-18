from scipy.io import loadmat
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pyod.utils.utility import standardizer
import numpy as np


def evaluate_single_split(X_train_std, X_test_std, Y_test):
    iforest = IForest()
    dif = DIF()
    loda = LODA()

    iforest.fit(X_train_std)
    dif.fit(X_train_std)
    loda.fit(X_train_std)

    iforest_scores = iforest.decision_function(X_test_std)
    y_pred_forest = (iforest_scores > iforest.threshold_).astype(int)

    dif_scores = dif.decision_function(X_test_std)
    y_pred_dif = (dif_scores > dif.threshold_).astype(int)

    loda_scores = loda.decision_function(X_test_std)
    y_pred_loda = (loda_scores > loda.threshold_).astype(int)

    results = {
        "iforest": (
            balanced_accuracy_score(Y_test, y_pred_forest),
            roc_auc_score(Y_test, iforest_scores),
        ),
        "dif": (
            balanced_accuracy_score(Y_test, y_pred_dif),
            roc_auc_score(Y_test, dif_scores),
        ),
        "loda": (
            balanced_accuracy_score(Y_test, y_pred_loda),
            roc_auc_score(Y_test, loda_scores),
        ),
    }
    return results


def main():
    # 1. Incarcam setul de date shuttle.mat, il impartim si il standardizam
    dataset = loadmat("shuttle.mat")
    print(dataset)
    X = dataset["X"]
    Y = dataset["y"].ravel()
    print("X shape:", X.shape, "Y shape:", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
    X_train_std, X_test_std = standardizer(X_train, X_test)

    # 2. Evaluare pe o singura impartire (train/test split)
    single_res = evaluate_single_split(X_train_std, X_test_std, Y_test)
    print("IForest ROC-AUC: ", single_res["iforest"][1])
    print("DIF ROC-AUC: ", single_res["dif"][1])
    print("LODA ROC-AUC: ", single_res["loda"][1])
    print("IForest Balanced Accuracy: ", single_res["iforest"][0])
    print("DIF Balanced Accuracy: ", single_res["dif"][0])
    print("LODA Balanced Accuracy: ", single_res["loda"][0])

    # 3. Calculam BA si ROC-AUC mediu pe 10 impartiri diferite ale setului de date
    ba_if, roc_auc_if = [], []
    ba_dif, roc_auc_dif = [], []
    ba_loda, roc_auc_loda = [], []

    for _ in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
        X_train_std, X_test_std = standardizer(X_train, X_test)

        res = evaluate_single_split(X_train_std, X_test_std, Y_test)
        ba_if.append(res["iforest"][0])
        roc_auc_if.append(res["iforest"][1])
        ba_dif.append(res["dif"][0])
        roc_auc_dif.append(res["dif"][1])
        ba_loda.append(res["loda"][0])
        roc_auc_loda.append(res["loda"][1])

    print("Mean Balanced Accuracy IForest: ", np.mean(ba_if))
    print("Mean ROC-AUC IForest: ", np.mean(roc_auc_if))
    print("Mean Balanced Accuracy DIF: ", np.mean(ba_dif))
    print("Mean ROC-AUC DIF: ", np.mean(roc_auc_dif))
    print("Mean Balanced Accuracy LODA: ", np.mean(ba_loda))
    print("Mean ROC-AUC LODA: ", np.mean(roc_auc_loda))


if __name__ == "__main__":
    main()
