"""
改进的 DLEM (Description-Level Embedding Module) 物理语义属性模块
将故障的物理特性（频率、幅值、调制模式等）编码为可学习的语义属性向量
用于零样本场景中将已知故障知识迁移到未知故障
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# 轴承故障物理属性定义
# ==============================================================================

# 标准故障物理属性矩阵（每行是一种故障类型，每列是一个物理属性）
# 属性包括：
#   1. 冲击周期性 (0-1)         - 故障是否产生周期性冲击
#   2. 频率范围 (0-1)            - 特征频率所在范围 (低/中/高)
#   3. 调幅程度 (0-1)            - 振幅调制强度
#   4. 调频程度 (0-1)            - 频率调制强度
#   5. 边频带丰富度 (0-1)        - 旁频带数量
#   6. 谐波衰减 (0-1)            - 高次谐波衰减速度
#   7. 载荷敏感度 (0-1)          - 对载荷变化的敏感程度
#   8. 转速敏感度 (0-1)          - 对转速变化的敏感程度
#   9. 内圈特征 (0-1)            - 是否与内圈相关
#  10. 外圈特征 (0-1)            - 是否与外圈相关
#  11. 滚动体特征 (0-1)          - 是否与滚动体相关
#  12. 保持架特征 (0-1)          - 是否与保持架相关
#  13. 冲击能量强度 (0-1)        - 单次冲击的能量大小
#  14. 宽带噪声增量 (0-1)        - 是否增加宽带噪声
#  15. 低频调制 (0-1)            - 低频包络调制
#  16. 复合故障概率 (0-1)        - 作为复合故障组分的概率

BEARING_FAULT_ATTRIBUTES = {
    # 故障类型: [16个物理属性]
    "Normal":    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    "IR_slight": [0.8, 0.6, 0.7, 0.3, 0.6, 0.4, 0.6, 0.7, 1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.7, 0.3],
    "IR_medium": [0.9, 0.7, 0.8, 0.5, 0.8, 0.5, 0.7, 0.8, 1.0, 0.0, 0.0, 0.0, 0.7, 0.5, 0.8, 0.5],
    "IR_severe": [1.0, 0.8, 0.9, 0.7, 0.9, 0.6, 0.8, 0.9, 1.0, 0.0, 0.0, 0.0, 0.9, 0.7, 0.9, 0.7],
    "OR_slight": [0.8, 0.5, 0.3, 0.2, 0.4, 0.3, 0.8, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.2, 0.3, 0.3],
    "OR_medium": [0.9, 0.6, 0.4, 0.3, 0.5, 0.4, 0.9, 0.6, 0.0, 1.0, 0.0, 0.0, 0.7, 0.4, 0.4, 0.5],
    "OR_severe": [1.0, 0.7, 0.5, 0.4, 0.7, 0.5, 1.0, 0.7, 0.0, 1.0, 0.0, 0.0, 0.9, 0.6, 0.5, 0.7],
    "B_slight":  [0.6, 0.8, 0.2, 0.6, 0.3, 0.5, 0.4, 0.9, 0.0, 0.0, 1.0, 0.0, 0.4, 0.4, 0.2, 0.2],
    "B_medium":  [0.7, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.6, 0.3, 0.4],
    "B_severe":  [0.8, 1.0, 0.4, 0.8, 0.6, 0.7, 0.6, 1.0, 0.0, 0.0, 1.0, 0.0, 0.8, 0.8, 0.4, 0.6],
    # 复合故障
    "IR_OR":     [1.0, 0.7, 0.8, 0.6, 0.9, 0.5, 0.8, 0.8, 1.0, 1.0, 0.0, 0.0, 0.8, 0.6, 0.8, 1.0],
    "IR_B":      [0.9, 0.8, 0.7, 0.7, 0.7, 0.5, 0.6, 0.9, 1.0, 0.0, 1.0, 0.0, 0.7, 0.6, 0.6, 1.0],
    "OR_B":      [0.9, 0.7, 0.4, 0.6, 0.6, 0.5, 0.8, 0.8, 0.0, 1.0, 1.0, 0.0, 0.7, 0.6, 0.4, 1.0],
    "IR_OR_B":   [1.0, 0.8, 0.8, 0.8, 0.9, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 0.0, 0.9, 0.8, 0.8, 1.0],
}


# ======================================================================
# SEU Gearbox (gearset) fault attributes
#
# These are lightweight, heuristic attributes aligned to the same 16-dim
# schema used for bearings. They are meant to provide a structured semantic
# signal for strict ZSL experiments on the SEU gearset classes.
#
# Class order (recommended): Health, Chipped, Miss, Root, Surface
# ======================================================================


SEU_GEAR_FAULT_ATTRIBUTES = {
    # Health working state
    "Health":    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    # Chipped tooth: periodic impacts, moderate modulation/sidebands
    "Chipped":   [0.7, 0.6, 0.5, 0.3, 0.6, 0.5, 0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.5, 0.3],
    # Missing tooth: stronger impacts and broadband noise, higher severity
    "Miss":      [0.9, 0.7, 0.7, 0.4, 0.8, 0.5, 0.7, 0.9, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.7, 0.5],
    # Root fault: impact + modulation, medium-high severity
    "Root":      [0.8, 0.6, 0.6, 0.4, 0.7, 0.5, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.7, 0.6, 0.6, 0.4],
    # Surface fault: tends to add broadband components and sidebands
    "Surface":   [0.6, 0.7, 0.4, 0.3, 0.7, 0.6, 0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.5, 0.7, 0.4, 0.3],
}


# ==========================================================================
# Baidu Gear Dataset (百度网盘齿轮数据集) fault attributes
#
# 5 种故障类型：健康, 剥落, 断齿, 磨损, 裂纹
# 16 维齿轮专用物理属性：
#   0. 冲击周期性 (0-1)        – 啮合频率处的冲击周期性强弱
#   1. 频谱能量集中度 (0-1)     – 频谱能量集中在啮合频率 vs 分散
#   2. 调幅深度 (0-1)          – 啮合频率幅值波动的强度
#   3. 调频深度 (0-1)          – 瞬时频率波动的强度
#   4. 边频带丰富度 (0-1)      – 啮合频率周围边频带的数量和幅值
#   5. 谐波丰富度 (0-1)        – 啮合频率谐波的数量 (高次谐波)
#   6. 载荷敏感度 (0-1)        – 对载荷变化的敏感程度
#   7. 转速敏感度 (0-1)        – 对转速变化的敏感程度
#   8. 局部损伤标记 (0-1)      – 齿面局部剥落/点蚀特征
#   9. 断裂/缺齿标记 (0-1)     – 断齿特有的大冲击+随动特征
#  10. 分布式磨损标记 (0-1)    – 齿面均匀磨损，摩擦增大
#  11. 裂纹标记 (0-1)          – 齿根裂纹导致的刚度周期性变化
#  12. 单次冲击能量 (0-1)      – 单次冲击的峰值强度
#  13. 宽带噪声增量 (0-1)      – 宽带噪声基底抬升幅度
#  14. 低频包络调制 (0-1)      – 由齿轮旋转一周带来的低频调制
#  15. 复合故障概率 (0-1)      – 保留字段（复合故障用）
# ==========================================================================

# BAIDU_GEAR_FAULT_ATTRIBUTES — 极值化齿轮专用 16 维属性
# 设计原则（借鉴轴承属性的成功经验）：
#   每类故障在 3-4 个专属维度上用 0.9-1.0，其余维度用 0.0-0.1
#   确保任意两个类别的余弦距离 > 0.6，语义空间高度可分
# 注意：复合故障概率 (dim 15) 保留为 0，BaiduGear 数据集无复合故障类
BAIDU_GEAR_FAULT_ATTRIBUTES = {
    # 健康：全部近零，只有工况维度有微弱信号
    "健康":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
    # 剥落：冲击周期性(0) + 边频带(4) + 局部损伤(8) → 三大特征拉满
    "剥落":  [0.95, 0.05, 0.1, 0.0, 0.9, 0.1,  0.4, 0.3,  0.95, 0.0, 0.0, 0.0, 0.5, 0.1, 0.2, 0.0],
    # 断齿：冲击周期性(0) + 调幅深度(2) + 断裂标记(9) + 冲击能量(12) → 四个特征极值
    "断齿":  [1.0,  0.0,  0.95,0.7, 0.2, 0.05, 0.3, 0.3,  0.0,  0.95,0.0, 0.0, 0.95,0.2, 0.4, 0.0],
    # 磨损：频率集中度(1)高 + 分布式磨损(10) + 宽带噪声(13) → 分布式损伤特征
    "磨损":  [0.1,  0.9,  0.05,0.0, 0.1, 0.1,  0.5, 0.4,  0.0,  0.0, 0.95,0.0, 0.1, 0.9, 0.1, 0.0],
    # 裂纹：调频深度(3) + 载荷敏感(6) + 裂纹标记(11) → 周期性刚度变化特征
    "裂纹":  [0.5,  0.1,  0.2, 0.8, 0.3, 0.1,  0.95,0.4,  0.1,  0.0, 0.0, 0.95,0.3, 0.1, 0.3, 0.0],
}


# Named attribute sets for different datasets.
ATTRIBUTE_SETS = {
    "bearing": BEARING_FAULT_ATTRIBUTES,
    "seu_gear": SEU_GEAR_FAULT_ATTRIBUTES,
    "baidu_gear": BAIDU_GEAR_FAULT_ATTRIBUTES,
}

# CWRU 数据集到标准属性的映射
CWRU_ATTRIBUTE_MAP = {
    "Normal":  "Normal",
    "IR007":   "IR_slight",
    "IR014":   "IR_medium",
    "IR021":   "IR_severe",
    "B007":    "B_slight",
    "B014":    "B_medium",
    "B021":    "B_severe",
    "OR007":   "OR_slight",
    "OR014":   "OR_medium",
    "OR021":   "OR_severe",
}


class PhysicalAttributeEncoder(nn.Module):
    """
    物理属性编码器：将离散的物理属性向量映射到连续的语义嵌入空间
    """

    def __init__(self, num_raw_attributes: int = 16,
                 attribute_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_raw_attributes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, attribute_dim),
            nn.LayerNorm(attribute_dim),
        )

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attributes: (B, num_raw_attributes) 物理属性向量
        Returns:
            embeddings: (B, attribute_dim) 语义嵌入
        """
        return self.encoder(attributes)


class AdaptiveAttributeRefiner(nn.Module):
    """
    自适应属性精炼器：利用视觉特征动态调整语义属性
    解决固定属性矩阵无法适应不同工况的问题
    """

    def __init__(self, visual_dim: int, attribute_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        # 视觉特征到调整因子
        self.visual_gate = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, attribute_dim),
            nn.Sigmoid()
        )
        # 视觉特征到偏移量
        self.visual_bias = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, attribute_dim),
            nn.Tanh()
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, semantic_attr: torch.Tensor,
                visual_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_attr: (B, attribute_dim) 基础语义属性
            visual_feat: (B, visual_dim) 视觉特征
        Returns:
            refined_attr: (B, attribute_dim) 精炼后的语义属性
        """
        gate = self.visual_gate(visual_feat)
        bias = self.visual_bias(visual_feat) * self.scale
        refined = semantic_attr * gate + bias
        return F.normalize(refined, p=2, dim=-1)


class SemanticAttributeModule(nn.Module):
    """
    改进的语义属性模块 (改进 DLEM)
    1. 物理属性编码：将手工定义的物理属性映射到高维语义空间
    2. 自适应精炼：利用视觉特征动态调整语义向量
    3. 跨模态对齐：确保视觉特征和语义属性在同一空间
    """

    def __init__(self, visual_dim: int = 768, attribute_dim: int = 128,
                 hidden_dim: int = 256, num_raw_attributes: int = 16,
                 fault_attributes: dict = None):
        super().__init__()
        self.attribute_dim = attribute_dim

        # 物理属性编码器
        self.attr_encoder = PhysicalAttributeEncoder(
            num_raw_attributes, attribute_dim, hidden_dim
        )

        # 自适应属性精炼器
        self.refiner = AdaptiveAttributeRefiner(
            visual_dim, attribute_dim, hidden_dim
        )

        # 视觉特征到语义空间的映射
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, attribute_dim),
        )

        # 注册故障属性矩阵
        if fault_attributes is None:
            fault_attributes = BEARING_FAULT_ATTRIBUTES
        self._build_attribute_matrix(fault_attributes)

        # Keep a (non-state_dict) copy for ablations / runtime switching.
        self._physical_attribute_matrix = self.attribute_matrix.detach().clone()

    def set_attribute_mode(self, mode: str = 'physical', *, seed: int = 0) -> None:
        """Switch raw attribute matrix for ablations.

        Notes:
        - Does NOT change any parameters; only overwrites the existing buffer
          in-place (checkpoint-compatible).
        - Keeps shape (num_classes, num_raw_attributes) unchanged.

        Modes:
        - physical: the original BEARING_FAULT_ATTRIBUTES matrix
        - onehot: a simple one-hot-like coding (first C dims), padded to num_raw_attributes
        - random: uniform[0,1] attributes (deterministic by seed)
        """
        m = str(mode).lower().strip() if mode is not None else 'physical'
        if m not in {'physical', 'onehot', 'random'}:
            raise ValueError(f"Unknown attribute mode: {mode}")

        base = self._physical_attribute_matrix
        device = self.attribute_matrix.device
        dtype = self.attribute_matrix.dtype
        C, K = int(base.shape[0]), int(base.shape[1])

        if m == 'physical':
            new_mat = base
        elif m == 'onehot':
            oh = torch.zeros((C, K), dtype=base.dtype)
            for i in range(C):
                if i < K:
                    oh[i, i] = 1.0
            new_mat = oh
        else:
            g = torch.Generator(device='cpu')
            g.manual_seed(int(seed))
            new_mat = torch.rand((C, K), generator=g, dtype=base.dtype)

        if new_mat.shape != self.attribute_matrix.shape:
            raise RuntimeError(
                f"Attribute matrix shape mismatch: {tuple(new_mat.shape)} vs {tuple(self.attribute_matrix.shape)}"
            )

        self.attribute_matrix.data.copy_(new_mat.to(device=device, dtype=dtype))

    def _build_attribute_matrix(self, fault_attributes: dict):
        """构建并注册故障属性矩阵"""
        self.fault_names = list(fault_attributes.keys())
        attr_matrix = torch.tensor(
            [fault_attributes[name] for name in self.fault_names],
            dtype=torch.float32
        )
        self.register_buffer('attribute_matrix', attr_matrix)

    def get_class_attributes(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        根据类别索引获取对应的物理属性向量
        Args:
            class_indices: (B,) 类别索引
        Returns:
            attributes: (B, num_raw_attributes)
        """
        return self.attribute_matrix[class_indices]

    def encode_attributes(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        编码类别的语义属性
        Args:
            class_indices: (B,) 类别索引
        Returns:
            semantic_embeddings: (B, attribute_dim)
        """
        raw_attrs = self.get_class_attributes(class_indices)
        return self.attr_encoder(raw_attrs)

    def forward(self, visual_features: torch.Tensor,
                class_indices: Optional[torch.Tensor] = None,
                raw_attributes: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: (B, visual_dim) Swin提取的视觉特征
            class_indices: (B,) 类别索引 (训练时使用)
            raw_attributes: (B, num_raw_attributes) 直接提供的属性 (生成时使用)
        Returns:
            dict containing:
                - semantic_embedding: (B, attribute_dim) 语义嵌入
                - visual_embedding: (B, attribute_dim) 视觉映射到语义空间
                - refined_embedding: (B, attribute_dim) 精炼后的语义嵌入
        """
        # 视觉特征映射到语义空间
        visual_embedding = self.visual_projector(visual_features)
        visual_embedding = F.normalize(visual_embedding, p=2, dim=-1)

        # 获取基础语义属性
        if raw_attributes is not None:
            base_semantic = self.attr_encoder(raw_attributes)
        elif class_indices is not None:
            base_semantic = self.encode_attributes(class_indices)
        else:
            raise ValueError("Must provide either class_indices or raw_attributes")

        # 自适应精炼
        refined_embedding = self.refiner(base_semantic, visual_features)

        return {
            'semantic_embedding': base_semantic,
            'visual_embedding': visual_embedding,
            'refined_embedding': refined_embedding,
        }

    def get_all_class_embeddings(self) -> torch.Tensor:
        """
        获取所有类别的语义嵌入（用于零样本推理）
        Returns:
            embeddings: (num_classes, attribute_dim)
        """
        all_indices = torch.arange(len(self.fault_names),
                                   device=self.attribute_matrix.device)
        return self.encode_attributes(all_indices)


class AttributeInterpolator:
    """
    属性插值器：为未知故障类型生成语义属性
    通过已知故障属性的组合/插值来推断未知故障的物理特性
    """

    @staticmethod
    def interpolate(attr1: np.ndarray, attr2: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
        """线性插值两种故障属性"""
        return attr1 * (1 - alpha) + attr2 * alpha

    @staticmethod
    def compose_compound_fault(attrs: List[np.ndarray],
                               weights: List[float] = None) -> np.ndarray:
        """
        组合多个单一故障属性为复合故障属性
        Args:
            attrs: 各单一故障的属性向量列表
            weights: 各故障的权重
        """
        if weights is None:
            weights = [1.0 / len(attrs)] * len(attrs)
        result = np.zeros_like(attrs[0])
        for attr, w in zip(attrs, weights):
            result += attr * w
        # 复合故障的复合概率设为1
        result[-1] = 1.0
        return np.clip(result, 0, 1)

    @staticmethod
    def perturb(attr: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """对属性添加扰动（模拟工况变化）"""
        noise = np.random.randn(*attr.shape) * noise_std
        return np.clip(attr + noise, 0, 1)
