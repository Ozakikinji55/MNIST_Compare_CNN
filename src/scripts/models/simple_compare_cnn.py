# models/simple_compare_cnn.py

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """A simple residual block"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # residual connection
        return self.relu(out)

class PowerfulEncoder(nn.Module):
    """Enhanced encoder using residual blocks"""
    def __init__(self, in_channels=1, base_channels=32, feat_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 64x64 -> 32x32
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResBlock(base_channels * 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResBlock(base_channels * 4)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.fc(x)
        return feat

class CompareNet(nn.Module):
    def __init__(self, feat_dim=128, scale=1.0, symmetric_aux=True):
        super().__init__()
        # using the powerful encoder
        self.encoder = PowerfulEncoder(in_channels=1, feat_dim=feat_dim)
        self.scale = scale
        self.symmetric_aux = symmetric_aux

        # Using a small MLP to enhance comparation
        self.comparator = nn.Sequential
        (
            nn.Linear(feat_dim * 2, feat_dim // 2), # Process the spliced ​​features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), #Add Dropout to prevent overfitting
            nn.Linear(feat_dim // 2, 1)
        )

        if self.symmetric_aux:
            # The auxiliary head is also enhanced
            self.aux_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 2, 1)
            )

    def forward(self, xa, xb):
        fa = self.encoder(xa)
        fb = self.encoder(xb)

        # Compare fa and fb
        combined_features = torch.cat([fa, fb], dim=1)
        main_logit = self.comparator(combined_features).squeeze(-1) * self.scale

        if self.symmetric_aux and self.training:
            # Auxiliary task: judging the "direction" of (fa-fb)
            diff_feat = fa - fb
            aux_logit = self.aux_head(diff_feat).squeeze(-1)
            return main_logit, aux_logit
        
        return main_logit


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

