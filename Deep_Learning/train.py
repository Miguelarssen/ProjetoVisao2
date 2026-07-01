import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from pathlib import Path

from dataset import get_dataloaders
from models.cnn3 import CNN3

def train_model():
    # Hiperparâmetros padrões
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Caminhos
    # Assumindo que este script rode de dentro de Deep_Learning,
    # a pasta Dataset_Originals e mlruns estão um nível acima
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "Dataset_Originals"
    MLRUNS_DIR = BASE_DIR / "Deep_Learning" / "mlruns"
    SAVED_MODELS_DIR = BASE_DIR / "saved_models"
    
    if not SAVED_MODELS_DIR.exists():
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Pasta de dados não encontrada: {DATA_DIR}")
        
    print("Carregando datasets...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE
    )
    
    num_classes = len(classes)
    print(f"Total de classes encontradas: {num_classes}")
    
    # Configuração do dispositivo (GPU se disponível)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Instanciando o modelo
    model = CNN3(num_classes=num_classes).to(device)
    
    # Loss e Otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Configurando MLflow
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("Projeto2_DeepLearning")
    
    with mlflow.start_run():
        # Registrando parâmetros
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("model", "CNN_3_Layers")
        
        print("Iniciando treinamento...")
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            start_time = time.time()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            
            # Validação
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Time: {epoch_time:.1f}s")
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
            
            # Registrando métricas por época no MLflow
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
            mlflow.log_metric("val_acc", val_epoch_acc, step=epoch)
            
        print("Treinamento finalizado. Iniciando teste no conjunto Test (10%)...")
        
        # Teste Final
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        test_acc = test_correct / test_total
        print(f"Acurácia final no conjunto de Teste: {test_acc:.4f}")
        mlflow.log_metric("test_acc", test_acc)
        
        # Registrando o modelo no MLflow
        mlflow.pytorch.log_model(model, "model")
        print("Modelo salvo no MLflow com sucesso.")
        
        # Salvando o modelo localmente na pasta .gitignore
        model_name = model.__class__.__name__
        model_save_path = SAVED_MODELS_DIR / f"{model_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Pesos do modelo salvos fisicamente em: {model_save_path}")

if __name__ == "__main__":
    train_model()
