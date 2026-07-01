"""
Arquivo de configuração do pipeline de Machine Learning.

Altere aqui o caminho do dataset e parâmetros gerais.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR / "Dataset_Originals")

IMAGE_SIZE = (128, 128)

# Use None para usar todas as imagens.
# Para testar rápido, use 300 ou 500.
MAX_IMAGES_PER_CLASS = None

TEST_SIZE = 0.20
RANDOM_STATE = 42
