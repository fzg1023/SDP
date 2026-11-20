"""实测 v10 最终模型的模块组成与参数量（只统计，不训练）。"""
from __future__ import annotations
import os, sys
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
from esag.utils import load_config
from esag.model import ESAG

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
ATTR_DIM = 21  # 图像域物理特征维度


def build():
    config = load_config(CFG)
    config["model"]["backbone_type"] = "swin_lite"
    config["model"]["use_generative"] = False
    config["model"]["semantic"]["num_attributes"] = ATTR_DIM
    config["model"]["semantic"]["attribute_set"] = "baidu_gear"
    config["model"]["swin"]["drop_path_rate"] = 0.3
    model = ESAG(config)
    fd = model.feature_dim
    model.attr_head = nn.Linear(fd, ATTR_DIM)
    return model, config


def count(module):
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    model, config = build()
    backbone = model.backbone
    head = model.attr_head

    print("=== 配置 ===")
    print("backbone_type =", config["model"]["backbone_type"])
    print("use_generative =", config["model"]["use_generative"])
    print("feature_dim =", model.feature_dim)
    print("attr_dim =", ATTR_DIM)
    print()

    print("=== 模块组成 ===")
    print("1) 预处理(输入前): 双通道图 (包络谱 | CWT幅度), 形状 (2,64,64)")
    print("2) 骨干 backbone (swin_lite): (2,64,64) -> 特征 (B, feature_dim)")
    print(f"   参数量 = {count(backbone):,}")
    print("3) 视觉投影器 attr_head (Linear): (B, feature_dim) -> (B, 21)")
    print(f"   参数量 = {count(head):,}")
    print("4) 分类器: 语义点积 argmax_c normalize(g)·normalize(a_c), a_c 为 21 维物理属性原型")
    print()

    total = count(backbone) + count(head)
    print(f"=== 总可训练参数量 = {total:,} ===")
    print(f"  其中骨干 {count(backbone):,} + 投影头 {count(head):,}")

    # 骨干内部分层粗看
    print("\n=== 骨干子模块参数量 ===")
    for name, sub in backbone.named_children():
        print(f"  {name}: {count(sub):,}")
