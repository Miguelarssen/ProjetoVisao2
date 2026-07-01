"""
Carregamento do dataset.

Este arquivo lê uma pasta única organizada por classes, por exemplo:

C:\Dataset_Originals2
├── Apple___Apple_scab
├── Apple___Black_rot
├── Apple___healthy
└── ...
"""

import os
import numpy as np

from config import MAX_IMAGES_PER_CLASS
from features import extrair_caracteristicas


def carregar_dataset_unico(pasta):
    X = []
    y = []

    if not os.path.exists(pasta):
        raise FileNotFoundError(f"A pasta não foi encontrada: {pasta}")

    classes = sorted(os.listdir(pasta))

    for classe in classes:
        caminho_classe = os.path.join(pasta, classe)

        if not os.path.isdir(caminho_classe):
            continue

        print(f"Carregando classe: {classe}")

        arquivos = sorted(os.listdir(caminho_classe))

        if MAX_IMAGES_PER_CLASS is not None:
            arquivos = arquivos[:MAX_IMAGES_PER_CLASS]

        total_classe = 0

        for arquivo in arquivos:
            caminho_imagem = os.path.join(caminho_classe, arquivo)

            try:
                caracteristicas = extrair_caracteristicas(caminho_imagem)
                X.append(caracteristicas)
                y.append(classe)
                total_classe += 1
            except Exception as erro:
                print(f"Erro ao processar {caminho_imagem}: {erro}")

        print(f"  Imagens carregadas: {total_classe}")

    return np.array(X), np.array(y)
