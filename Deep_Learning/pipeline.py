import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from pathlib import Path
import numpy as np

# Importando o Dataset e as Arquiteturas
from dataset import get_dataloaders
from models.cnn3 import CNN3
from models.cnn5 import CNN5
from models.resnet import CustomResNet18
from models.convnext import CustomConvNeXtTiny
from models.efficientnet import CustomEfficientNetB0

def run_experiment(model_class, model_name, dataloaders, classes, device, base_dir, epochs=10, batch_size=32, lr=0.001):
    train_loader, val_loader, test_loader = dataloaders
    num_classes = len(classes)
    
    # Prepara o diretório de modelos salvos
    SAVED_MODELS_DIR = base_dir / "Deep_Learning" / "saved_models"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Inicializando o modelo
    model = model_class(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n{'='*50}\nIniciando Experimento: {model_name}\n{'='*50}")
    
    with mlflow.start_run(run_name=model_name):
        # Registrando parâmetros
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("model", model_name)
        
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
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
            val_loss, val_correct, val_total = 0.0, 0, 0
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
            
            print(f"Epoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
            
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
            mlflow.log_metric("val_acc", val_epoch_acc, step=epoch)
            
        print("\nTestando o modelo no conjunto final...")
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculando Métricas Globais (Scikit-Learn)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Test Accuracy a partir das listas
        test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("f1_score_macro", f1)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        
        # Matriz de Confusão Visual
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        os.remove(cm_filename) # Limpa arquivo temporário
        
        # Registrando o modelo no MLflow
        mlflow.pytorch.log_model(model, "model")
        
        # Salvando fisicamente
        model_save_path = SAVED_MODELS_DIR / f"{model_name}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Experimento finalizado! Pesos salvos em: {model_save_path}\n")

def run_pipeline():
    # Caminhos
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "Dataset_Originals"
    MLRUNS_DIR = BASE_DIR / "Deep_Learning" / "mlruns"
    
    print("Carregando datasets uma única vez para todos os modelos...")
    # Carrega dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=DATA_DIR, 
        batch_size=32
    )
    dataloaders = (train_loader, val_loader, test_loader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Configura MLflow Globalmente
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("Projeto2_DeepLearning")
    
    # Definição dos modelos a serem iterados
    models_to_train = [
        (CNN3, "CNN3"),
        (CNN5, "CNN5"),
        (CustomResNet18, "ResNet18"),
        (CustomConvNeXtTiny, "ConvNeXtTiny"),
        (CustomEfficientNetB0, "EfficientNetB0")
    ]
    
    for model_class, model_name in models_to_train:
        # Você pode alterar as épocas aqui
        run_experiment(
            model_class=model_class,
            model_name=model_name,
            dataloaders=dataloaders,
            classes=classes,
            device=device,
            base_dir=BASE_DIR,
            epochs=10, 
            batch_size=32,
            lr=0.001
        )

if __name__ == "__main__":
    run_pipeline()
