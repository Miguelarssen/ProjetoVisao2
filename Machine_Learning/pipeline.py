"""
Pipeline principal de Machine Learning.

Execute este arquivo:

python pipeline.py

Etapas:
1. Carrega o dataset limpo
2. Divide em treino e validação
3. Treina Random Forest, KNN e SVM
4. Compara resultados
5. Mostra a matriz de confusão do melhor modelo
"""

import os
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from pathlib import Path

from config import DATA_DIR, TEST_SIZE, RANDOM_STATE
from dataset_loader import carregar_dataset_unico
from models import obter_modelos
from evaluate import avaliar_modelo, plotar_matriz_confusao


def main():
    print("\n==============================")
    print("CARREGANDO DATASET ORIGINAL LIMPO")
    print("==============================")

    X, y = carregar_dataset_unico(DATA_DIR)

    print("\nResumo geral:")
    print("X:", X.shape)
    print("y:", y.shape)

    print("\nClasses encontradas:")
    for classe in sorted(set(y)):
        print("-", classe)

    print("\nTotal de imagens:", len(y))

    print("\n==============================")
    print("DIVIDINDO DATASET")
    print("==============================")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_valid:", X_valid.shape)
    print("y_valid:", y_valid.shape)

    modelos = obter_modelos()

    resultados = {}
    melhor_modelo_nome = None
    melhor_acuracia = 0
    melhor_y_pred = None

    # Configuração do MLflow
    BASE_DIR = Path(__file__).resolve().parent
    MLRUNS_DIR = BASE_DIR / "mlruns_ml"
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("Projeto2_MachineLearning")

    for nome_modelo, modelo in modelos.items():
        print("\n==============================")
        print(f"TREINANDO MODELO: {nome_modelo}")
        print("==============================")

        with mlflow.start_run(run_name=nome_modelo):
            modelo.fit(X_train, y_train)

            print("Treinamento concluído.")

            acuracia, precisao, recall, f1, y_pred = avaliar_modelo(
                nome_modelo,
                modelo,
                X_valid,
                y_valid
            )

            # Registrando parâmetros
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("model_name", nome_modelo)
            
            # Registrando parâmetros específicos do modelo, se possível
            try:
                mlflow.log_params(modelo.get_params())
            except Exception:
                pass

            # Registrando métricas
            mlflow.log_metric("acuracia", acuracia)
            mlflow.log_metric("precisao_macro", precisao)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("f1_macro", f1)

            # Salvando e registrando matriz de confusão
            cm_filename = plotar_matriz_confusao(y_valid, y_pred, nome_modelo)
            mlflow.log_artifact(cm_filename)
            os.remove(cm_filename) # Limpa o arquivo local

            # Salvando o modelo
            mlflow.sklearn.log_model(modelo, "model")

            resultados[nome_modelo] = acuracia

            if acuracia > melhor_acuracia:
                melhor_acuracia = acuracia
                melhor_modelo_nome = nome_modelo
                melhor_y_pred = y_pred

    print("\n==============================")
    print("RESUMO COMPARATIVO")
    print("==============================")

    for nome_modelo, acuracia in resultados.items():
        print(f"{nome_modelo}: {acuracia:.4f}")

    print(f"\nMelhor modelo: {melhor_modelo_nome}")
    print(f"Melhor acurácia: {melhor_acuracia:.4f}")

    print("\n==============================")
    print("FINALIZADO")
    print("==============================")
    print("Pipeline executado com sucesso e logado no MLflow (mlruns_ml).")


if __name__ == "__main__":
    main()
