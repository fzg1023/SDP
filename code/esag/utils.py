"""
ESAG 工具函数模块
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional, Set


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典：override 覆盖 base（不修改入参）。"""
    merged: Dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge_dict(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def load_config(config_path: str, _visited: Optional[Set[str]] = None) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    支持在配置中使用 `extends:` 继承另一个配置文件，然后对其进行覆盖：

    - extends 支持相对路径（相对于当前 config 文件所在目录）或绝对路径
    - 当前配置会覆盖 base 配置（深度合并）
    """
    config_path = os.path.abspath(config_path)
    if _visited is None:
        _visited = set()
    if config_path in _visited:
        raise ValueError(f"Config extends cycle detected: {config_path}")
    _visited.add(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    extends_path = config.pop('extends', None)
    if not extends_path:
        return config

    if not os.path.isabs(extends_path):
        extends_path = os.path.join(os.path.dirname(config_path), extends_path)

    base_config = load_config(extends_path, _visited=_visited)
    return _deep_merge_dict(base_config, config)


def set_seed(seed: int = 42):
    """设置全局随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config: str = "auto") -> torch.device:
    """自动检测可用设备"""
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_config)


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path, extra=None):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if extra:
        checkpoint.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device='cpu'):
    """加载训练检查点"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 20, min_delta: float = 0.0,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
