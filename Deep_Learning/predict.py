import sys
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from models.cnn3 import CNN3
from dataset import get_dataloaders

def predict_image(image_path):
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "Dataset_Originals"
    MODEL_PATH = BASE_DIR / "saved_models" / "CNN3.pth"
    
    if not MODEL_PATH.exists():
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return
        
    if not Path(image_path).exists():
        print(f"Erro: Imagem não encontrada em {image_path}")
        return

    # Pega os nomes das classes diretamente das pastas (ou vc pode fixar a lista)
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    num_classes = len(classes)
    
    # 1. Carrega o modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN3(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Prepara a imagem (Mesmas transforms de validação)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # Adiciona dimensão do batch
    
    # 3. Faz a predição
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = classes[predicted_idx.item()]
    conf_percent = confidence.item() * 100
    
    print(f"\n--- Resultado da Classificação ---")
    print(f"Imagem: {image_path}")
    print(f"Classe Prevista: {predicted_class}")
    print(f"Confiança: {conf_percent:.2f}%")
    print("-" * 32 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py <caminho_para_imagem.jpg>")
    else:
        predict_image(sys.argv[1])
