# utils.py
import logging, random, os
import numpy as np
import torch
from typing import Tuple, Optional

# ============== 全局 device / logger ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("brain_completion")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

def set_seed(seed: int = 42):
    """固定随机种子，保证可复现实验。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_global_normalization_params(train_dataset: dict, num_rois: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐 ROI 汇总所有样本的时间序列，计算全局 mean/std。
    train_dataset: {sid: {'session_1': {'rest_features': (n_rois, T), ...}}}
    返回: (num_rois,), (num_rois,)
    """
    roi_data = [[] for _ in range(num_rois)]
    for subj_data in train_dataset.values():
        features = subj_data['session_1']['rest_features']
        assert features.shape[0] == num_rois, f"ROI数量不匹配: 预期{num_rois}，实际{features.shape[0]}"
        for roi_idx in range(num_rois):
            roi_time_series = features[roi_idx, :].flatten()
            roi_data[roi_idx].extend(roi_time_series.tolist())

    global_mean = np.zeros(num_rois)
    global_std  = np.zeros(num_rois)
    for roi_idx in range(num_rois):
        data = np.array(roi_data[roi_idx])
        global_mean[roi_idx] = data.mean() if data.size else 0.0
        global_std[roi_idx]  = data.std()  if data.size else 1.0
    return global_mean, global_std

class DynamicMaskGenerator:
    """基于 ROI 粒度的随机掩蔽，用于模拟缺失 ROI。"""
    def __init__(self, mask_rate: Optional[float] = None):
        self.mask_rate = mask_rate if mask_rate is not None else 0.3

    def generate_mask(self, shape, mode: Optional[str] = None) -> torch.Tensor:
        """
        根据输入张量形状生成布尔掩码：
          - shape=(n_rois, T)
          - shape=(B, n_rois, T)
          - shape=(B, C, n_rois, T)
        True 表示“被掩蔽（缺失）”的位置。
        """
        if mode is None:
            mode = 'roi'
        if mode != 'roi':
            raise ValueError(f"无效的掩蔽模式: {mode}")

        if len(shape) == 2:
            n_rois = shape[0]
            mask = torch.zeros(shape, dtype=torch.bool)
            roi_mask = torch.rand(n_rois) < self.mask_rate
            mask[roi_mask, :] = True
            return mask

        elif len(shape) == 3:
            B, n_rois, T = shape
            mask = torch.zeros(shape, dtype=torch.bool)
            for i in range(B):
                roi_mask = torch.rand(n_rois) < self.mask_rate
                mask[i, roi_mask, :] = True
            return mask

        elif len(shape) == 4:
            B, C, n_rois, T = shape
            mask = torch.zeros(shape, dtype=torch.bool)
            for i in range(B):
                roi_mask = torch.rand(n_rois) < self.mask_rate
                mask[i, :, roi_mask, :] = True
            return mask

        else:
            raise ValueError("Unsupported tensor shape for ROI mask generation.")
# utils.py
import logging, random, os
import numpy as np
import torch
from typing import Tuple, Optional

# ============== 全局 device / logger ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("brain_completion")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

def set_seed(seed: int = 42):
    """固定随机种子，保证可复现实验。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_global_normalization_params(train_dataset: dict, num_rois: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐 ROI 汇总所有样本的时间序列，计算全局 mean/std。
    train_dataset: {sid: {'session_1': {'rest_features': (n_rois, T), ...}}}
    返回: (num_rois,), (num_rois,)
    """
    roi_data = [[] for _ in range(num_rois)]
    for subj_data in train_dataset.values():
        features = subj_data['session_1']['rest_features']
        assert features.shape[0] == num_rois, f"ROI数量不匹配: 预期{num_rois}，实际{features.shape[0]}"
        for roi_idx in range(num_rois):
            roi_time_series = features[roi_idx, :].flatten()
            roi_data[roi_idx].extend(roi_time_series.tolist())

    global_mean = np.zeros(num_rois)
    global_std  = np.zeros(num_rois)
    for roi_idx in range(num_rois):
        data = np.array(roi_data[roi_idx])
        global_mean[roi_idx] = data.mean() if data.size else 0.0
        global_std[roi_idx]  = data.std()  if data.size else 1.0
    return global_mean, global_std

class DynamicMaskGenerator:
    """基于 ROI 粒度的随机掩蔽，用于模拟缺失 ROI。"""
    def __init__(self, mask_rate: Optional[float] = None):
        self.mask_rate = mask_rate if mask_rate is not None else 0.3

    def generate_mask(self, shape, mode: Optional[str] = None) -> torch.Tensor:
        """
        根据输入张量形状生成布尔掩码：
          - shape=(n_rois, T)
          - shape=(B, n_rois, T)
          - shape=(B, C, n_rois, T)
        True 表示“被掩蔽（缺失）”的位置。
        """
        if mode is None:
            mode = 'roi'
        if mode != 'roi':
            raise ValueError(f"无效的掩蔽模式: {mode}")

        if len(shape) == 2:
            n_rois = shape[0]
            mask = torch.zeros(shape, dtype=torch.bool)
            roi_mask = torch.rand(n_rois) < self.mask_rate
            mask[roi_mask, :] = True
            return mask

        elif len(shape) == 3:
            B, n_rois, T = shape
            mask = torch.zeros(shape, dtype=torch.bool)
            for i in range(B):
                roi_mask = torch.rand(n_rois) < self.mask_rate
                mask[i, roi_mask, :] = True
            return mask

        elif len(shape) == 4:
            B, C, n_rois, T = shape
            mask = torch.zeros(shape, dtype=torch.bool)
            for i in range(B):
                roi_mask = torch.rand(n_rois) < self.mask_rate
                mask[i, :, roi_mask, :] = True
            return mask

        else:
            raise ValueError("Unsupported tensor shape for ROI mask generation.")
