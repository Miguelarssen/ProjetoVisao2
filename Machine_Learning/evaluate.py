"""
Funções de avaliação dos modelos.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


def avaliar_modelo(nome_modelo, modelo, X_valid, y_valid):
    print("\n==============================")
    print(f"AVALIAÇÃO DO MODELO: {nome_modelo}")
    print("==============================")

    y_pred = modelo.predict(X_valid)

    acuracia = accuracy_score(y_valid, y_pred)
    precisao = precision_score(y_valid, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_valid, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_valid, y_pred, average="macro", zero_division=0)

    print(f"\nAcurácia do {nome_modelo}: {acuracia:.4f}")

    print("\nRelatório de classificação:")
    print(classification_report(y_valid, y_pred))

    return acuracia, precisao, recall, f1, y_pred


def plotar_matriz_confusao(y_valid, y_pred, nome_modelo):
    classes_ordenadas = sorted(set(y_valid))

    matriz = confusion_matrix(
        y_valid,
        y_pred,
        labels=classes_ordenadas
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        xticklabels=classes_ordenadas,
        yticklabels=classes_ordenadas
    )

    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f"confusion_matrix_{nome_modelo}.png"
    plt.savefig(filename)
    plt.close() # Fecha a figura para não consumir memória
    return filename
