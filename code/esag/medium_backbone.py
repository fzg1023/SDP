"""
中等规模 CNN 骨干（Medium CNN Backbone，基于 ResNet-18）

介于 SmallCNN (~2M) 与 Swin-Tiny+CBAM (27.6M) 之间：用标准 ResNet-18 (~11.6M)
替换 SmallCNN，以恢复 SmallCNN 在「语义点积 easy 划分」上丢失的判别能力，
同时保持远小于 Swin 的参数量。

输入仍为双通道（包络谱 + CWT）64×64 图，输出与 Swin 骨干对齐的 768 维特征。
"""

import torch
import torch.nn as nn


class MediumBackbone(nn.Module):
    """ResNet-18 骨干，~11.6M 参数，输出 out_dim 维特征。"""

    def __init__(self, out_dim: int = 768):
        super().__init__()
        self.out_dim = int(out_dim)

        try:
            from torchvision.models import resnet18
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "MediumBackbone requires torchvision. Install with: pip install torchvision"
            ) from e

        self.backbone = resnet18(num_classes=self.out_dim)

        # 双通道输入：把 ResNet 默认的 3 通道 conv1 换成 2 通道。
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            2, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # 用原 3 通道权重的均值初始化 2 通道权重，加速收敛。
        with torch.no_grad():
            w = old_conv.weight.mean(dim=1, keepdim=True)  # (out, 1, k, k)
            new_conv.weight.copy_(w.repeat(1, 2, 1, 1))
        self.backbone.conv1 = new_conv

        self.num_features = self.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def set_cbam_enabled(self, enabled: bool) -> None:
        """与 SwinBackbone 保持相同接口（ResNet 无 CBAM，空操作）。"""
        return None
