import torch.nn as nn
import torch.nn.functional as F

class CNN3(nn.Module):
    def __init__(self, num_classes):
        super(CNN3, self).__init__()
        
        # Bloco Convolucional 1
        # Entrada: (3, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloco Convolucional 2
        # Entrada: (32, 64, 64)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloco Convolucional 3
        # Entrada: (64, 32, 32)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Após 3 pools de 2x2, a imagem 128x128 vira 16x16
        # Flattening size: 128 (channels) * 16 (height) * 16 (width) = 32768
        
        # Camadas Fully Connected (Lineares)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Passando pelos blocos conv
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Achatar (flatten)
        x = x.view(x.size(0), -1)
        
        # Passando pelas camadas fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
