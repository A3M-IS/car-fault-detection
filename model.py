import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class EngineCRNN(nn.Module):
    def __init__(self, num_classes):
        super(EngineCRNN, self).__init__()
        
        # Initial Conv
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual Blocks with SE
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2) # Increased to 512
        
        self.spatial_dropout = nn.Dropout2d(0.25)

        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=config.LSTM_HIDDEN_SIZE, 
            num_layers=config.LSTM_LAYERS, 
            batch_first=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0,
            bidirectional=True
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE * 2, config.LSTM_HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(config.LSTM_HIDDEN_SIZE, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE * 2, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, 1, 128, 128)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.spatial_dropout(x)
        
        # Pool frequency bins
        x = torch.mean(x, dim=2)
        x = x.permute(0, 2, 1).contiguous()
        
        # LSTM
        lstm_out, _ = self.lstm(x) 

        # Attention pooling
        attention_logits = self.temporal_attention(lstm_out)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.classifier(pooled)
        return out

def build_model(num_classes):
    return EngineCRNN(num_classes)
