"""
Definição dos modelos clássicos de Machine Learning usados no experimento.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def obter_modelos():
    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1,
            gamma="scale"
        )
    }

    return modelos
