import numpy as np
from numpy import histogram, linalg as la
from numpy.random import multivariate_normal, uniform
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def compute_histograms_and_probs(proj, n_bins=10, value_range=(-5, 5)):
    histograms = []
    bin_edges = []
    for i in range(proj.shape[1]):
        hist, edges = histogram(proj[:, i], range=value_range, bins=n_bins)
        histograms.append(hist)
        bin_edges.append(edges)

    probs = [hist / proj.shape[0] for hist in histograms]
    return histograms, bin_edges, probs


def compute_anomaly_scores(X, v, histograms, bin_edges, probs):
    anomaly_scores = []
    for x in X:
        proj_x = x @ v.T
        s = 0.0
        for i in range(len(proj_x)):
            for b in range(len(histograms[i])):
                if bin_edges[i][b] <= proj_x[i] < bin_edges[i][b + 1]:
                    s += probs[i][b]
                    break
        anomaly_scores.append(s / len(histograms))
    return anomaly_scores


def plot_scores(X_test, scores, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="coolwarm")
    plt.colorbar(label="Anomaly Score")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def main():
    # 1. Generam datasetul cu make_blobs
    X_train, Y_train = make_blobs(n_samples=500, n_features=2)
    print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)

    # 2. Generam cei 5 vectori unitate de proiectie
    mean = [0, 0]
    cov = np.eye(2)
    print("Covarianta:\n", cov)

    v = multivariate_normal(mean, cov, 5)
    print("v shape:", v.shape)

    norm = la.norm(v, axis=1, keepdims=True)
    v = v / norm
    print("Vectori unitate v:\n", v)

    # 3. Calculam histogramele si probabilitatile (default: 10 bin-uri)
    proj = X_train @ v.T
    print("proj shape:", proj.shape)

    histograms, bin_edges, probs = compute_histograms_and_probs(proj, n_bins=10, value_range=(-5, 5))
    print("Histograms (10 bins):", histograms)
    print("Bin edges (10 bins):", bin_edges)
    print("Probs (10 bins):", probs)

    # 4. Calculam anomaly score pentru fiecare esantion din X_train
    anomaly_scores_train = compute_anomaly_scores(X_train, v, histograms, bin_edges, probs)
    print("Anomaly scores train:", anomaly_scores_train)

    # 5. Testare: Generam 500 de puncte distribuite uniform intre [-3, 3]
    X_test = uniform(-3, 3, (500, 2))
    print("X_test shape:", X_test.shape)

    anomaly_scores_test = compute_anomaly_scores(X_test, v, histograms, bin_edges, probs)
    print("Anomaly scores test (10 bins):", anomaly_scores_test)
    plot_scores(X_test, anomaly_scores_test, "Anomaly Scores for Test Data (10 Bins)")

    # 6. Schimbam numarul de bins la 5
    histograms_5, bin_edges_5, probs_5 = compute_histograms_and_probs(proj, n_bins=5, value_range=(-5, 5))
    print("Probs (5 bins):", probs_5)
    anomaly_scores_test_5 = compute_anomaly_scores(X_test, v, histograms_5, bin_edges_5, probs_5)
    print("Anomaly scores test (5 bins):", anomaly_scores_test_5)
    plot_scores(X_test, anomaly_scores_test_5, "Anomaly Scores for Test Data with 5 Bins")

    # 7. Schimbam numarul de bins la 50
    histograms_50, bin_edges_50, probs_50 = compute_histograms_and_probs(proj, n_bins=50, value_range=(-5, 5))
    print("Probs (50 bins):", probs_50)
    anomaly_scores_test_50 = compute_anomaly_scores(X_test, v, histograms_50, bin_edges_50, probs_50)
    print("Anomaly scores test (50 bins):", anomaly_scores_test_50)
    plot_scores(X_test, anomaly_scores_test_50, "Anomaly Scores for Test Data with 50 Bins")


if __name__ == "__main__":
    main()
