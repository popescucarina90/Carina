"""
Detectarea anomaliilor în setul Cardio folosind KNN, LOF și combinarea scorurilor.

Fișierul cardio.mat trebuie să se afle în același director cu acest script.
"""

import numpy as np
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.combination import average, maximization


# ---------------------------------------------------------
# 1. Încărcarea și împărțirea datelor
# ---------------------------------------------------------

date = sio.loadmat("cardio.mat")
X = np.asarray(date["X"], dtype=float)
y = np.asarray(date["y"]).ravel().astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("Dimensiunea setului de antrenare:", X_train.shape)
print("Dimensiunea setului de testare:", X_test.shape)

# Standardizarea se învață pe setul de antrenare.
X_train_scaled, X_test_scaled = standardizer(X_train, X_test)


# ---------------------------------------------------------
# 2. Antrenarea modelelor pentru mai multe valori k
# ---------------------------------------------------------

k_values = np.arange(30, 121, 9)

knn_train_scores = []
knn_test_scores = []
lof_train_scores = []
lof_test_scores = []

for k in k_values:
    # Evită alegerea unui număr de vecini prea mare.
    k_current = min(int(k), len(X_train_scaled) - 1)

    print(f"\nNumăr de vecini: {k_current}")

    model_knn = KNN(n_neighbors=k_current)
    model_lof = LOF(n_neighbors=k_current)

    model_knn.fit(X_train_scaled)
    model_lof.fit(X_train_scaled)

    # Scorurile de anomalia pentru datele de antrenare.
    knn_train_scores.append(model_knn.decision_scores_)
    lof_train_scores.append(model_lof.decision_scores_)

    # Scorurile pentru datele de testare.
    knn_test_scores.append(model_knn.decision_function(X_test_scaled))
    lof_test_scores.append(model_lof.decision_function(X_test_scaled))


# ---------------------------------------------------------
# 3. Construirea matricilor de scoruri și normalizarea lor
# ---------------------------------------------------------

knn_train_matrix = np.column_stack(knn_train_scores)
knn_test_matrix = np.column_stack(knn_test_scores)

lof_train_matrix = np.column_stack(lof_train_scores)
lof_test_matrix = np.column_stack(lof_test_scores)

knn_train_norm, knn_test_norm = standardizer(
    knn_train_matrix, knn_test_matrix
)
lof_train_norm, lof_test_norm = standardizer(
    lof_train_matrix, lof_test_matrix
)


# ---------------------------------------------------------
# 4. Combinarea scorurilor: media și maximul
# ---------------------------------------------------------

knn_avg_train = average(knn_train_norm)
knn_avg_test = average(knn_test_norm)

knn_max_train = maximization(knn_train_norm)
knn_max_test = maximization(knn_test_norm)

lof_avg_train = average(lof_train_norm)
lof_avg_test = average(lof_test_norm)

lof_max_train = maximization(lof_train_norm)
lof_max_test = maximization(lof_test_norm)


# ---------------------------------------------------------
# 5. Stabilirea pragurilor pe baza setului de antrenare
# ---------------------------------------------------------

contamination = 0.096
quantile_level = 1.0 - contamination

thresholds = {
    "KNN - maxim": np.quantile(knn_max_train, quantile_level),
    "KNN - medie": np.quantile(knn_avg_train, quantile_level),
    "LOF - maxim": np.quantile(lof_max_train, quantile_level),
    "LOF - medie": np.quantile(lof_avg_train, quantile_level),
}

score_sets = {
    "KNN - maxim": (knn_max_train, knn_max_test),
    "KNN - medie": (knn_avg_train, knn_avg_test),
    "LOF - maxim": (lof_max_train, lof_max_test),
    "LOF - medie": (lof_avg_train, lof_avg_test),
}


# ---------------------------------------------------------
# 6. Evaluarea rezultatelor
# ---------------------------------------------------------

print("\n========== REZULTATE ==========")

for method, (scores_train, scores_test) in score_sets.items():
    threshold = thresholds[method]

    predictions_train = (scores_train >= threshold).astype(int)
    predictions_test = (scores_test >= threshold).astype(int)

    train_ba = balanced_accuracy_score(y_train, predictions_train)
    test_ba = balanced_accuracy_score(y_test, predictions_test)

    print(f"\nMetoda: {method}")
    print(f"Prag: {threshold:.5f}")
    print(f"Balanced accuracy - antrenare: {train_ba:.4f}")
    print(f"Balanced accuracy - testare:   {test_ba:.4f}")
