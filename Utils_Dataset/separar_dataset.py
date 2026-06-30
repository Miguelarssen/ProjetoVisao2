import os
import shutil
import re
from pathlib import Path

def separar_originais(src_dir, dest_dir):
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    # Cria a pasta de destino (limpa se já existir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Regex para identificar os sufixos de Data Augmentation usados no dataset
    aug_pattern = re.compile(
        r'_(flipLR|flipTB|180deg|90deg|270deg|new30degFlipLR|new30degFlipTB|newGRR|newPixel25|new90degFlipTB|newGGR|new200degFlipTB|new200degFlipLR|new90degFlipLR|new[A-Za-z0-9]+)$'
    )

    copiados = 0
    ignorados = 0

    # Percorre as pastas train e valid do dataset original
    for split in ['train', 'valid']:
        split_dir = src_dir / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            dest_class_dir = dest_dir / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in class_dir.iterdir():
                if img_path.is_file():
                    # Verifica se o nome do arquivo (sem a extensão .JPG) possui o sufixo de augmentation
                    if aug_pattern.search(img_path.stem):
                        ignorados += 1
                        continue # Ignora a imagem aumentada
                    
                    # Se for original, copia para a nova pasta
                    dest_path = dest_class_dir / img_path.name
                    if not dest_path.exists():
                        shutil.copy2(img_path, dest_path)
                        copiados += 1

    print(f"Extração concluída!")
    print(f"Imagens originais copiadas: {copiados}")
    print(f"Imagens com augmentation ignoradas: {ignorados}")

if __name__ == "__main__":
    # Caminhos para o dataset
    PASTA_ORIGEM = r"Dataset_MachineLearning\Dataset_MachineLearning"
    PASTA_DESTINO = r"Dataset_Originals"
    
    separar_originais(PASTA_ORIGEM, PASTA_DESTINO)
