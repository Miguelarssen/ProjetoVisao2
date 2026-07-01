import torch.nn as nn
from torchvision import models

class CustomConvNeXtTiny(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNeXtTiny, self).__init__()
        
        # Carrega o ConvNeXt Tiny sem pesos pré-treinados
        self.model = models.convnext_tiny(weights=None)
        
        # A cabeça de classificação do ConvNeXt fica em classifier[2]
        # É uma camada Linear(768, 1000)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
