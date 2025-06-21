import torch
import torch.nn as nn
from torchvision import models
import timm  # Để sử dụng các mô hình khác nếu cần

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]
        fpn_outs = []
        last_inner = laterals[-1]
        fpn_outs.append(self.fpn_convs[-1](last_inner))
        for i in range(len(laterals) - 2, -1, -1):
            top_down = nn.functional.interpolate(last_inner, scale_factor=2, mode='nearest')
            lateral = laterals[i]
            last_inner = lateral + top_down
            fpn_outs.insert(0, self.fpn_convs[i](last_inner))
        return fpn_outs

class MiniTransformer(nn.Module):
    def __init__(self, embed_dim=192, num_patches=49, num_heads=4, num_layers=2, dropout=0.1):
        super(MiniTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Tầng Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Tầng đầu ra
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch_size, num_patches, embed_dim]
        x = self.transformer(x)  # [batch_size, num_patches, embed_dim]
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        x = self.output_layer(x)  # [batch_size, embed_dim]
        return x

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ChestXrayModel, self).__init__()
        # EfficientNet-B3 backbone
        self.efficientnet = models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT' if pretrained else None)
        self.efficientnet.classifier = nn.Identity()  # Bỏ classifier mặc định
        
        # Lấy các tầng feature từ EfficientNet-B3
        self.features = nn.ModuleList([
            self.efficientnet.features[2],  # Stage 2: độ phân giải 56x56
            self.efficientnet.features[3],  # Stage 3: độ phân giải 28x28
            self.efficientnet.features[4],  # Stage 4: độ phân giải 14x14
            self.efficientnet.features[6],  # Stage 6: độ phân giải 7x7
        ])
        
        # FPN để hợp nhất các feature map
        in_channels_list = [32, 48, 96, 232]  # Số kênh thực tế của các stage trong EfficientNet-B3
        self.fpn = FPN(in_channels_list, out_channels=256)
        
        # Chuẩn bị feature map cho mini-Transformer
        self.conv_reducer = nn.Conv2d(256, 192, kernel_size=1)  # Giảm số kênh từ 256 xuống 192
        self.transformer = MiniTransformer(embed_dim=192, num_patches=49, num_heads=4, num_layers=2, dropout=0.1)
        
        # Nhánh meta-data
        self.metadata_fc = nn.Sequential(
            nn.Linear(3, 64),  # 3 đặc trưng: tuổi, giới tính, góc chụp
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Đầu ra phân loại
        self.classifier = nn.Sequential(
            nn.Linear(192 + 128, 512),  # Kết hợp feature từ Transformer và meta-data
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # Nhánh localization (heatmap)
        self.heatmap_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, metadata=None):
        # Trích xuất feature từ EfficientNet-B3
        features = []
        for i, layer in enumerate(self.efficientnet.features):
            x = layer(x)
            if i in [2, 3, 4, 6]:
                features.append(x)
        
        # FPN
        fpn_outs = self.fpn(features)
        
        # Lấy feature map cuối cùng và chuẩn bị cho mini-Transformer
        x = fpn_outs[-1]  # [batch_size, 256, 7, 7]
        x = self.conv_reducer(x)  # [batch_size, 192, 7, 7]
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [batch_size, 49, 192]
        
        # Mini-Transformer
        feature_out = self.transformer(x)  # [batch_size, 192]
        
        # Xử lý meta-data
        if metadata is not None:
            metadata_out = self.metadata_fc(metadata)  # (B, 128)
            combined = torch.cat([feature_out, metadata_out], dim=1)  # (B, 192+128)
        else:
            combined = torch.cat([feature_out, torch.zeros(feature_out.size(0), 128).to(feature_out.device)], dim=1)
        
        # Đầu ra phân loại
        logits = self.classifier(combined)
        
        # Đầu ra heatmap cho localization
        heatmap = self.heatmap_conv(fpn_outs[-1])  # (B, num_classes, H, W)
        
        return logits, heatmap
    
class ChestXraySegModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ChestXraySegModel, self).__init__()
        self.efficientnet = models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT' if pretrained else None)
        
        self.features = nn.ModuleList([
            self.efficientnet.features[2],
            self.efficientnet.features[3],
            self.efficientnet.features[4],
            self.efficientnet.features[6],
        ])
        
        in_channels_list = [32, 48, 96, 232]
        self.fpn = FPN(in_channels_list, out_channels=256)
        
        self.seg_conv = nn.Conv2d(256 + num_classes, num_classes, kernel_size=1)

    def forward(self, x, heatmap=None):
        features = []
        for i, layer in enumerate(self.efficientnet.features):
            x = layer(x)
            if i in [2, 3, 4, 6]:
                features.append(x)
        
        fpn_outs = self.fpn(features)
        x = fpn_outs[-1]
        
        if heatmap is not None:
            heatmap = nn.functional.interpolate(heatmap, size=(7, 7), mode='bilinear', align_corners=False)
            x = torch.cat([x, heatmap], dim=1)
        
        seg_mask = self.seg_conv(x)
        seg_mask = nn.functional.interpolate(seg_mask, size=(224, 224), mode='bilinear', align_corners=False)
        seg_mask = torch.sigmoid(seg_mask)
        return seg_mask

def get_model(model_name='chestxray_model', num_classes=5, pretrained=True):
    if model_name == 'chestxray_model':
        model = ChestXrayModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model

def get_seg_model(model_name='chestxray_seg_model', num_classes=5, pretrained=True):
    if model_name == 'chestxray_seg_model':
        model = ChestXraySegModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model