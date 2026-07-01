import torch.nn as nn
from torchvision import models

class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetB0, self).__init__()
        
        # Carrega o EfficientNet B0 sem pesos pré-treinados
        self.model = models.efficientnet_b0(weights=None)
        
        # A cabeça de classificação do EfficientNet fica em classifier[1]
        # É uma camada Linear(1280, 1000)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
