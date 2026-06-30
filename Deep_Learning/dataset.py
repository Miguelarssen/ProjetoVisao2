import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PlantDiseaseDataset(Dataset):
    """Dataset customizado para carregar as imagens de plantas e seus rótulos."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(data_dir, batch_size=32, img_size=(128, 128), random_state=42):
    """
    Retorna os dataloaders para treino (80%), validação (10%) e teste (10%).
    Aplica Data Augmentation apenas no treino.
    """
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_paths = []
    all_labels = []
    
    for cls_name in classes:
        cls_dir = data_dir / cls_name
        for img_path in cls_dir.iterdir():
            if img_path.is_file():
                all_paths.append(str(img_path))
                all_labels.append(class_to_idx[cls_name])
                
    # Stratified Split
    # Divisão 1: 80% treino, 20% resto (validação e teste)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.20, random_state=random_state, stratify=all_labels
    )
    
    # Divisão 2: do resto, 50% validação (10% total) e 50% teste (10% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    # Transformações (Data Augmentation apenas no treino)
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Instanciando Datasets
    train_dataset = PlantDiseaseDataset(X_train, y_train, transform=train_transforms)
    val_dataset = PlantDiseaseDataset(X_val, y_val, transform=val_test_transforms)
    test_dataset = PlantDiseaseDataset(X_test, y_test, transform=val_test_transforms)
    
    # Instanciando DataLoaders
    # num_workers=0 no Windows costuma ser mais estável
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Total de imagens: Treino ({len(train_dataset)}), Validação ({len(val_dataset)}), Teste ({len(test_dataset)})")
    
    return train_loader, val_loader, test_loader, classes
