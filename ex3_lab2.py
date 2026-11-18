import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF


# Construirea unui set de date cu doua grupuri
date, etichete_reale = make_blobs(
    n_samples=[200, 100],
    n_features=2,
    center_box=(-10, 10),
    cluster_std=[2, 6],
    random_state=42
)

valori_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

for k in valori_k:
    detector_knn = KNN(contamination=0.07, n_neighbors=k)
    detector_lof = LOF(contamination=0.07, n_neighbors=k)

    detector_knn.fit(date)
    detector_lof.fit(date)

    rezultate_knn = detector_knn.predict(date)
    rezultate_lof = detector_lof.predict(date)

    figura, axe = plt.subplots(1, 2, figsize=(12, 5))

    axe[0].scatter(
        date[:, 0], date[:, 1],
        c=rezultate_knn, cmap="coolwarm", s=25
    )
    axe[0].set_title(f"Detectie KNN (k={k})")
    axe[0].set_xlabel("Axa X")
    axe[0].set_ylabel("Axa Y")

    axe[1].scatter(
        date[:, 0], date[:, 1],
        c=rezultate_lof, cmap="coolwarm", s=25
    )
    axe[1].set_title(f"Detectie LOF (k={k})")
    axe[1].set_xlabel("Axa X")
    axe[1].set_ylabel("Axa Y")

    figura.suptitle("Compararea metodelor de detectare a anomaliilor")
    figura.tight_layout()
    plt.show()
