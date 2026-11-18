import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data


def calculeaza_leverage(valori):
    """Calculeaza scorurile leverage pentru matricea primita."""
    matrice_hat = valori @ np.linalg.pinv(valori.T @ valori) @ valori.T
    return np.diag(matrice_hat)


def experimenteaza(nr_caracteristici):
    X, _, _, _ = generate_data(
        n_train=1000,
        n_test=0,
        contamination=0.1,
        n_features=nr_caracteristici
    )
    print("Dimensiunea setului:", X.shape)

    medie = float(input("Media zgomotului: "))
    varianta = float(input("Varianta zgomotului: "))
    abatere = np.sqrt(varianta)
    zgomot = np.random.normal(medie, abatere, size=(1000, 1))

    np.random.shuffle(X)
    grupuri = list(np.array_split(X, 4))
    grupuri[1] = grupuri[1] + np.random.normal(
        medie, abatere, size=grupuri[1].shape
    )
    grupuri[3] = grupuri[3] + np.random.normal(
        medie, abatere, size=grupuri[3].shape
    )

    panta = float(input("Valoarea pantei a: "))
    intercept = float(input("Valoarea termenului liber b: "))

    raspunsuri = []
    for pozitie, grup in enumerate(grupuri):
        factor = 10 if pozitie in (2, 3) else 1
        raspunsuri.append(
            panta * grup + intercept
            + np.random.normal(medie, factor * abatere, size=grup.shape)
        )

    X_complet = np.vstack(grupuri)
    if nr_caracteristici == 1:
        coeficient = np.random.uniform(-10, 10)
        liber = np.random.uniform(-10, 10)
        valori_model = coeficient * X_complet + liber + zgomot
    else:
        coef = np.random.uniform(-10, 10, size=nr_caracteristici)
        liber = np.random.uniform(-10, 10)
        valori_model = X_complet @ coef + liber + zgomot

    scoruri = calculeaza_leverage(valori_model)
    limita = np.quantile(scoruri, 0.95)
    masca_grupuri = np.array_split(scoruri >= limita, 4)

    print("Scorurile leverage:")
    print(scoruri)

    titluri = [
        "Date fara modificari",
        "Zgomot adaugat pe X",
        "Zgomot mai mare pe Y",
        "Zgomot pe X si Y"
    ]

    fig, axe = plt.subplots(2, 2, figsize=(11, 8))
    for index, ax in enumerate(axe.flat):
        ax.scatter(grupuri[index], raspunsuri[index], alpha=0.65)
        ax.scatter(
            grupuri[index][masca_grupuri[index]],
            raspunsuri[index][masca_grupuri[index]],
            color="crimson",
            label="Leverage ridicat"
        )
        ax.set_title(titluri[index])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    fig.suptitle(f"Analiza leverage pentru {nr_caracteristici} caracteristica(i)")
    fig.tight_layout()
    plt.show()


experimenteaza(1)
experimenteaza(2)
