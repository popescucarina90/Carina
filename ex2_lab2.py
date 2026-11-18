import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score


# Generarea datelor pentru antrenare si testare
X_train, X_test, y_train, y_test = generate_data_clusters(
    n_features=2,
    n_train=400,
    n_test=200,
    n_clusters=2,
    contamination=0.1
)


def deseneaza_rezultate(model, vecini, predictii_train, predictii_test):
    fig, axe = plt.subplots(2, 2, figsize=(11, 8))

    configuratii = [
        (axe[0, 0], X_train, y_train, "Antrenare - etichete reale"),
        (axe[0, 1], X_train, predictii_train, "Antrenare - predictii KNN"),
        (axe[1, 0], X_test, y_test, "Testare - etichete reale"),
        (axe[1, 1], X_test, predictii_test, "Testare - predictii KNN")
    ]

    for ax, date, etichete, titlu in configuratii:
        ax.scatter(date[:, 0], date[:, 1], c=etichete, cmap="coolwarm")
        ax.set_title(titlu)
        ax.set_xlabel("Caracteristica 1")
        ax.set_ylabel("Caracteristica 2")

    fig.suptitle(f"Clasificare KNN: k = {vecini}")
    fig.tight_layout()
    plt.show()


# Comparatia initiala folosind configuratia implicita
model_initial = KNN()
model_initial.fit(X_train)
predictii_train = model_initial.predict(X_train)
predictii_test = model_initial.predict(X_test)
deseneaza_rezultate(model_initial, model_initial.n_neighbors,
                    predictii_train, predictii_test)


# Evaluarea mai multor valori pentru numarul de vecini
lista_k = [1, 3, 5, 7, 9, 13, 15, 17, 19, 21]

for k in lista_k:
    clasificator = KNN(n_neighbors=k)
    clasificator.fit(X_train)

    estimari_train = clasificator.predict(X_train)
    estimari_test = clasificator.predict(X_test)

    scor_train = balanced_accuracy_score(y_train, estimari_train)
    scor_test = balanced_accuracy_score(y_test, estimari_test)

    print(f"\nNumar vecini: {k}")
    print(f"Acuratete echilibrata - train: {scor_train:.4f}")
    print(f"Acuratete echilibrata - test:  {scor_test:.4f}")

    deseneaza_rezultate(clasificator, k, estimari_train, estimari_test)
