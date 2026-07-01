import torch.nn as nn
from torchvision import models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        
        # Carrega a ResNet18 original (não pré-treinada ou pode ser pré-treinada)
        # Vamos inicializar sem pré-treino pesado para justificar o aprendizado rápido
        self.resnet = models.resnet18(weights=None)
        
        # Substituindo a última camada (Fully Connected) que originalmente prevê 1000 classes
        # Para o nosso número específico de classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
