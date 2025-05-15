import timm
import torch.nn as nn
from .utils import Res2TSMBlock, Res2NetBlock, TemporalShift


class MobileNetV4_Base(nn.Module):
    """
    Base MobileNetV4 Conv Blur Medium backbone with a simple pooling + classification head.
    """
    def __init__(self, model_key: str, dropout: float = 0.3):
        super().__init__()
        # Load pretrained MobileNetV4 Conv Blur Medium (features only)
        self.backbone = timm.create_model(model_key, pretrained=True, features_only=True)
        C = self.backbone.feature_info.channels()[-1]
        # Pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(C, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract final feature map
        feat = self.backbone(x)[-1]
        return self.forward_from_feat(feat)

    def forward_from_feat(self, feat):
        # Pool and classify from feature map
        out = self.global_pool(feat).view(feat.size(0), -1)
        return self.fc(out).squeeze(1)


class MobileNetV4_TSM(MobileNetV4_Base):
    """
    MobileNetV4 with a Temporal Shift Module injected at the final feature map.
    """
    def __init__(self, model_key: str, shift_div: int = 8, dropout: float = 0.3):
        super().__init__(model_key, dropout)
        C = self.backbone.feature_info.channels()[-1]
        self.temporal_shift = TemporalShift(C, shift_div)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        # Apply temporal shift
        feat = self.temporal_shift(feat)
        return self.forward_from_feat(feat)


class MobileNetV4_Res2Net(MobileNetV4_Base):
    """
    MobileNetV4 with a Res2Net-inspired multi-scale block at the final feature map.
    """
    def __init__(self, model_key: str, scale: int = 4, dropout: float = 0.3):
        super().__init__(model_key, dropout)
        C = self.backbone.feature_info.channels()[-1]
        self.res2net = Res2NetBlock(C, scale)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        # Apply Res2Net split-and-fuse
        feat = self.res2net(feat)
        return self.forward_from_feat(feat)


class MobileNetV4_Res2TSM(MobileNetV4_Base):
    """
    MobileNetV4 with combined Res2Net + Temporal Shift (Res2TSM) at the final feature map.
    """
    def __init__(self, model_key: str, scale: int = 4, shift_div: int = 8, dropout: float = 0.3):
        super().__init__(model_key, dropout)
        C = self.backbone.feature_info.channels()[-1]
        self.res2tsm = Res2TSMBlock(C, scale, shift_div)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        # Apply Res2TSM
        feat = self.res2tsm(feat)
        return self.forward_from_feat(feat)
