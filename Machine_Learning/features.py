"""
Funções para leitura das imagens e extração de características.

Características usadas:
- Histograma de cores RGB
- LBP (Local Binary Pattern)
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from config import IMAGE_SIZE


def ler_imagem(caminho_imagem):
    """
    Lê uma imagem usando np.fromfile + cv2.imdecode.
    Essa forma evita problemas no Windows com caminhos contendo acentos.
    """
    dados = np.fromfile(caminho_imagem, dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    return imagem


def extrair_caracteristicas(caminho_imagem):
    """
    Recebe o caminho de uma imagem e retorna um vetor numérico
    com características de cor e textura.
    """

    imagem = ler_imagem(caminho_imagem)

    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho_imagem}")

    imagem = cv2.resize(imagem, IMAGE_SIZE)

    # OpenCV lê em BGR, então convertemos para RGB
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # Histograma de cores RGB
    hist_r = cv2.calcHist([imagem_rgb], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([imagem_rgb], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([imagem_rgb], [2], None, [32], [0, 256]).flatten()

    hist_cor = np.concatenate([hist_r, hist_g, hist_b])
    hist_cor = hist_cor / (np.sum(hist_cor) + 1e-7)

    # LBP para textura
    imagem_cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)

    raio = 1
    pontos = 8 * raio

    lbp = local_binary_pattern(
        imagem_cinza,
        P=pontos,
        R=raio,
        method="uniform"
    )

    hist_lbp, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, pontos + 3),
        range=(0, pontos + 2)
    )

    hist_lbp = hist_lbp.astype("float")
    hist_lbp = hist_lbp / (hist_lbp.sum() + 1e-7)

    # 96 características de cor + 10 de textura = 106 características
    caracteristicas = np.concatenate([hist_cor, hist_lbp])

    return caracteristicas
