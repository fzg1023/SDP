"""
ESAG 信号预处理模块
包络分析 + 连续小波变换 (CWT) 双通道特征增强
"""

import numpy as np
import torch
from scipy.signal import hilbert, butter, filtfilt, resample
from scipy.ndimage import zoom
import pywt


class EnvelopeAnalyzer:
    """
    包络分析器：对振动信号做 Hilbert 包络解调
    放大微弱的周期性故障冲击（轴承点蚀、裂纹），抑制噪声
    """

    def __init__(self, sample_rate: int = 12000, filter_order: int = 5,
                 low_freq: float = 500.0, high_freq: float = None):
        """
        Args:
            sample_rate: 采样率 (Hz)
            filter_order: 带通滤波器阶数
            low_freq: 带通滤波低频截止 (Hz)
            high_freq: 带通滤波高频截止 (Hz), None则为奈奎斯特频率的0.9
        """
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.low_freq = low_freq
        self.high_freq = high_freq or (sample_rate * 0.45)

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """带通滤波，提取故障频段"""
        nyq = self.sample_rate / 2
        low = self.low_freq / nyq
        high = min(self.high_freq / nyq, 0.99)
        b, a = butter(self.filter_order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def envelope(self, signal: np.ndarray) -> np.ndarray:
        """Hilbert 包络解调"""
        # 带通滤波
        filtered = self.bandpass_filter(signal)
        # Hilbert 变换取包络
        analytic_signal = hilbert(filtered)
        envelope = np.abs(analytic_signal)
        return envelope

    def envelope_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """包络谱：对包络信号做 FFT，提取故障特征频率"""
        env = self.envelope(signal)
        # 去均值
        env = env - np.mean(env)
        # FFT
        n = len(env)
        spectrum = np.abs(np.fft.rfft(env)) / n
        return spectrum[:n // 2]


class CWTProcessor:
    """
    连续小波变换处理器：将 1D 振动信号转为 2D 时频图
    """

    def __init__(self, num_scales: int = 64, wavelet: str = 'morl',
                 output_size: int = 64):
        """
        Args:
            num_scales: 小波尺度数
            wavelet: 小波基函数 (morl, cmor, mexh 等)
            output_size: 输出图像尺寸
        """
        self.num_scales = num_scales
        self.wavelet = wavelet
        self.output_size = output_size
        self.scales = np.arange(1, num_scales + 1)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号做 CWT，得到时频系数矩阵
        
        Returns:
            coefficients: (num_scales, signal_length) 的 CWT 系数
        """
        coefficients, _ = pywt.cwt(signal, self.scales, self.wavelet)
        return np.abs(coefficients)

    def to_image(self, signal: np.ndarray) -> np.ndarray:
        """
        将 CWT 系数 resize 为固定大小的图像
        
        Returns:
            image: (output_size, output_size) 归一化图像
        """
        coefs = self.transform(signal)
        # 使用 zoom 调整为目标大小
        zoom_factors = (self.output_size / coefs.shape[0],
                        self.output_size / coefs.shape[1])
        image = zoom(coefs, zoom_factors, order=1)
        # 归一化到 [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image


class SignalPreprocessor:
    """
    信号预处理流水线：包络谱 + CWT 双通道特征增强
    输出: (2, H, W) 的双通道图像，用于 Swin Transformer 输入
    """

    def __init__(
        self,
        signal_length: int = 2048,
        sample_rate: int = 12000,
        filter_order: int = 5,
        num_scales: int = 64,
        cwt_wavelet: str = 'morl',
        output_size: int = 64,
        normalize: bool = True,
        channel_mode: str = 'dual',
    ):
        """
        Args:
            signal_length: 输入信号长度
            sample_rate: 采样率
            filter_order: 包络滤波器阶数
            num_scales: CWT 尺度数
            cwt_wavelet: CWT 小波基
            output_size: 输出图像大小
            normalize: 是否做 Z-score 归一化
        """
        self.signal_length = signal_length
        self.output_size = output_size
        self.normalize = normalize

        # Preprocessing ablations (paper-friendly): keep output shape (2,H,W)
        # so existing 2-channel backbones/checkpoints remain loadable.
        # - 'dual': envelope + CWT
        # - 'env_only': envelope only (CWT channel set to 0)
        # - 'cwt_only': CWT only (envelope channel set to 0)
        self.channel_mode = str(channel_mode).lower().strip() if channel_mode is not None else 'dual'
        if self.channel_mode not in {'dual', 'env_only', 'cwt_only'}:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")

        self.envelope_analyzer = EnvelopeAnalyzer(
            sample_rate=sample_rate,
            filter_order=filter_order
        )
        self.cwt_processor = CWTProcessor(
            num_scales=num_scales,
            wavelet=cwt_wavelet,
            output_size=output_size
        )

    def _prepare_signal(self, signal: np.ndarray) -> np.ndarray:
        """信号预处理：截断/补零 + 归一化"""
        # 截断或补零到固定长度
        if len(signal) > self.signal_length:
            signal = signal[:self.signal_length]
        elif len(signal) < self.signal_length:
            signal = np.pad(signal, (0, self.signal_length - len(signal)),
                            mode='constant')

        # Z-score 归一化
        if self.normalize:
            mean = np.mean(signal)
            std = np.std(signal) + 1e-8
            signal = (signal - mean) / std

        return signal

    def _envelope_to_image(self, signal: np.ndarray) -> np.ndarray:
        """
        将包络谱转为 2D 图像
        方法: 将 1D 包络谱 reshape 为 2D，然后 resize 到目标大小
        """
        env_spectrum = self.envelope_analyzer.envelope_spectrum(signal)

        # 将 1D 频谱 reshape 为近似正方形
        n = len(env_spectrum)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.pad(env_spectrum, (0, side * side - n), mode='constant')
        image_2d = padded.reshape(side, side)

        # Resize 到目标大小
        zoom_factors = (self.output_size / side, self.output_size / side)
        image = zoom(image_2d, zoom_factors, order=1)

        # 归一化到 [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image

    def process_single(self, signal: np.ndarray) -> np.ndarray:
        """
        处理单个信号，输出双通道图像
        
        Args:
            signal: (signal_length,) 原始振动信号
            
        Returns:
            image: (2, output_size, output_size) 双通道图像
                   通道0: 包络谱图像
                   通道1: CWT 时频图像
        """
        signal = self._prepare_signal(signal)

        mode = self.channel_mode

        if mode == 'env_only':
            env_image = self._envelope_to_image(signal)
            cwt_image = np.zeros_like(env_image)
        elif mode == 'cwt_only':
            cwt_image = self.cwt_processor.to_image(signal)
            env_image = np.zeros_like(cwt_image)
        else:
            # dual
            env_image = self._envelope_to_image(signal)
            cwt_image = self.cwt_processor.to_image(signal)

        dual_channel = np.stack([env_image, cwt_image], axis=0)
        return dual_channel.astype(np.float32)

    def process_batch(self, signals: np.ndarray) -> torch.Tensor:
        """
        批量处理信号
        
        Args:
            signals: (batch_size, signal_length) 批量信号
            
        Returns:
            images: (batch_size, 2, output_size, output_size) 批量双通道图像
        """
        batch = []
        for sig in signals:
            batch.append(self.process_single(sig))
        return torch.from_numpy(np.stack(batch, axis=0))

    def process_with_augmentation(self, signal: np.ndarray,
                                  noise_level: float = 0.01,
                                  shift_range: int = 50) -> np.ndarray:
        """
        带数据增强的处理：加噪声 + 随机位移
        
        Args:
            signal: 原始信号
            noise_level: 高斯噪声标准差
            shift_range: 随机循环位移范围
        """
        # 加高斯噪声
        if noise_level > 0:
            noise = np.random.randn(len(signal)) * noise_level
            signal = signal + noise

        # 随机循环位移
        if shift_range > 0:
            shift = np.random.randint(-shift_range, shift_range)
            signal = np.roll(signal, shift)

        return self.process_single(signal)
