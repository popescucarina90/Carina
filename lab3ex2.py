from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from numpy.random import uniform
import matplotlib.pyplot as plt


def plot_2d_comparison(X_test, scores_dif, scores_loda, scores_iforest, suptitle=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = [
        ("DIF", scores_dif),
        ("LODA", scores_loda),
        ("Isolation Forest", scores_iforest),
    ]
    for ax, (name, scores) in zip(axes, models):
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="coolwarm")
        fig.colorbar(sc, ax=ax, label="Anomaly Score")
        ax.set_title(f"Anomaly Scores for Test Data using {name}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_3d_comparison(X_test, scores_dif, scores_loda, scores_iforest, suptitle=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})
    models = [
        ("DIF", scores_dif),
        ("LODA", scores_loda),
        ("Isolation Forest", scores_iforest),
    ]
    for ax, (name, scores) in zip(axes, models):
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], scores, c=scores, cmap="coolwarm")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Anomaly Score")
        ax.set(title=f"Anomaly Scores ({name})", xlabel="x1", ylabel="x2", zlabel="Score")
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def main():
    # ==================== CAZUL 2D ====================
    # 1. Generam datele de antrenare cu 2 clustere
    X_train, Y_train = make_blobs(
        n_samples=[500, 500],
        centers=[(10, 0), (0, 10)],
        cluster_std=1,
    )

    # 2. Antrenam IForest si generam datele de test
    iforest = IForest(contamination=0.02)
    iforest.fit(X_train)
    X_test = uniform(-10, 20, size=(1000, 2))

    # Plot individual pentru IForest
    anomaly_scores_iforest = iforest.decision_function(X_test)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_scores_iforest, cmap="coolwarm")
    plt.colorbar(label="Anomaly Score")
    plt.title("Anomaly Scores for Test Data (IForest)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    # 3. Antrenam DIF si LODA (parametri default) si comparam in 3 subplots
    dif = DIF(contamination=0.02)
    dif.fit(X_train)
    anomaly_scores_dif = dif.decision_function(X_test)

    loda = LODA(contamination=0.02)
    loda.fit(X_train)
    anomaly_scores_loda = loda.decision_function(X_test)

    plot_2d_comparison(
        X_test,
        anomaly_scores_dif,
        anomaly_scores_loda,
        anomaly_scores_iforest,
        suptitle="2D Comparison (Default Parameters)",
    )

    # 4. Schimbam numarul de neuroni la DIF ([500, 5]) si numarul de bin-uri la LODA (5)
    dif_2 = DIF(hidden_neurons=[500, 5], contamination=0.02)
    dif_2.fit(X_train)
    anomaly_scores_dif_2 = dif_2.decision_function(X_test)

    loda_2 = LODA(n_bins=5, contamination=0.02)
    loda_2.fit(X_train)
    anomaly_scores_loda_2 = loda_2.decision_function(X_test)

    plot_2d_comparison(
        X_test,
        anomaly_scores_dif_2,
        anomaly_scores_loda_2,
        anomaly_scores_iforest,
        suptitle="2D Comparison (DIF hidden_neurons=[500, 5], LODA n_bins=5)",
    )

    # ==================== CAZUL 3D ====================
    X_train_3d, Y_train_3d = make_blobs(
        n_samples=[500, 500],
        n_features=3,
        centers=[(0, 10, 0), (10, 0, 10)],
        cluster_std=1,
    )
    X_test_3d = uniform(-10, 20, size=(1000, 3))

    iforest_3d = IForest(contamination=0.02)
    iforest_3d.fit(X_train_3d)
    scores_iforest_3d = iforest_3d.decision_function(X_test_3d)

    dif_3d = DIF(contamination=0.02)
    dif_3d.fit(X_train_3d)
    scores_dif_3d = dif_3d.decision_function(X_test_3d)

    loda_3d = LODA(contamination=0.02)
    loda_3d.fit(X_train_3d)
    scores_loda_3d = loda_3d.decision_function(X_test_3d)

    plot_3d_comparison(
        X_test_3d,
        scores_dif_3d,
        scores_loda_3d,
        scores_iforest_3d,
        suptitle="3D Comparison (Default Parameters)",
    )

    # 5. Schimbam numarul de neuroni si bin-uri in cazul 3D
    dif_3d_2 = DIF(hidden_neurons=[500, 5], contamination=0.02)
    dif_3d_2.fit(X_train_3d)
    scores_dif_3d_2 = dif_3d_2.decision_function(X_test_3d)

    loda_3d_2 = LODA(n_bins=5, contamination=0.02)
    loda_3d_2.fit(X_train_3d)
    scores_loda_3d_2 = loda_3d_2.decision_function(X_test_3d)

    plot_3d_comparison(
        X_test_3d,
        scores_dif_3d_2,
        scores_loda_3d_2,
        scores_iforest_3d,
        suptitle="3D Comparison (DIF hidden_neurons=[500, 5], LODA n_bins=5)",
    )


if __name__ == "__main__":
    main()
