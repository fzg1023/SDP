"""
SDP 骨干模型（backbone only）。

论文方法只用到卷积骨干（small CNN ≈1.96M 为主，Swin-Lite / ResNet-18 / Swin-Tiny
用于骨干消融）。生成/判别/语义对齐/KNN 等早期 ESAG 模块已移除，与论文描述一致。
"""
import torch.nn as nn

from .swin_backbone import SwinBackbone
from .small_backbone import SmallCNNBackbone
from .medium_backbone import MediumBackbone


class ESAG(nn.Module):
    """骨干封装：根据 backbone_type 选择特征提取骨干，暴露 backbone 与 feature_dim。"""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']
        swin_cfg = model_cfg['swin']

        backbone_type = str(model_cfg.get('backbone_type', 'small')).strip().lower()
        if backbone_type in ('small', 'medium'):
            # 输出维度与 Swin 骨干保持一致 (embed_dim * 2^(num_stages-1))
            out_dim = int(swin_cfg['embed_dim']) * (2 ** (len(swin_cfg['depths']) - 1))
            if backbone_type == 'medium':
                self.backbone = MediumBackbone(out_dim=out_dim)
            else:
                self.backbone = SmallCNNBackbone(out_dim=out_dim)
        elif backbone_type == 'swin':
            self.backbone = SwinBackbone(
                img_size=swin_cfg['img_size'],
                patch_size=swin_cfg['patch_size'],
                in_chans=swin_cfg['in_chans'],
                embed_dim=swin_cfg['embed_dim'],
                depths=swin_cfg['depths'],
                num_heads=swin_cfg['num_heads'],
                window_size=swin_cfg['window_size'],
                mlp_ratio=swin_cfg['mlp_ratio'],
                drop_rate=swin_cfg['drop_rate'],
                attn_drop_rate=swin_cfg['attn_drop_rate'],
                drop_path_rate=swin_cfg.get('drop_path_rate', 0.0),
            )
        elif backbone_type == 'swin_lite':
            lite_cfg = model_cfg.get('swin_lite', {}) or {}
            self.backbone = SwinBackbone(
                img_size=swin_cfg['img_size'],
                patch_size=swin_cfg['patch_size'],
                in_chans=swin_cfg['in_chans'],
                embed_dim=int(lite_cfg.get('embed_dim', 64)),
                depths=list(lite_cfg.get('depths', [2, 2, 4, 2])),
                num_heads=list(lite_cfg.get('num_heads', [2, 4, 8, 16])),
                window_size=swin_cfg['window_size'],
                mlp_ratio=swin_cfg['mlp_ratio'],
                drop_rate=swin_cfg['drop_rate'],
                attn_drop_rate=swin_cfg['attn_drop_rate'],
                drop_path_rate=swin_cfg.get('drop_path_rate', 0.0),
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type!r} "
                             f"(expected 'swin', 'swin_lite', 'medium', or 'small')")

        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)
