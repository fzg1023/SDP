"""
轻量 CNN 骨干（Small CNN Backbone）

用于替代 Swin-Tiny + CBAM 骨干：5 个卷积块 + 全局平均池化 + 线性头，
输出与 Swin 骨干对齐的 feature_dim 维特征（默认 768）。

骨干规模消融（supplement/run_backbone_ablation*.py）已证明：在本任务的
零样本设定下，该 ~2M 骨干与 27.6M 的 Swin-Tiny 骨干性能处于同一量级，
骨干容量并非性能的决定性因素。
"""

import torch
import torch.nn as nn


class SmallCNNBackbone(nn.Module):
    """~2M 参数的轻量卷积骨干，输出与 SwinBackbone 相同维度的特征向量。"""

    def __init__(self, out_dim: int = 768):
        super().__init__()
        self.out_dim = int(out_dim)
        channels = [32, 64, 128, 256, 512]
        layers = []
        cin = 2  # 双通道输入：包络谱 + CWT
        for cout in channels:
            layers += [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[-1], self.out_dim)
        self.num_features = self.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 512, 2, 2)
        x = self.pool(x)              # (B, 512, 1, 1)
        x = torch.flatten(x, 1)       # (B, 512)
        return self.head(x)           # (B, out_dim)

    def set_cbam_enabled(self, enabled: bool) -> None:
        """与 SwinBackbone 保持相同接口（无 CBAM，空操作）。"""
        return None
