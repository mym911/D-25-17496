# config.py
from pathlib import Path

class ModelConfig:
    def __init__(self):
        self.lr = 1e-4
        self.epochs_pretrain = 100
        self.epochs_gan = 200
        self.batch_size = 8
        self.mask_rate = 0.1
        self.cond_dim = 2  # [age, sex]

class DatasetConfig:
    def __init__(self):
        # 按你的机器修改这些路径
        self.phenotypic_csv = Path(r"E:/Users/数据集/ADHD200/adhd200_preprocessed_phenotypics.csv")
        self.roots = {
            "train": Path(r"E:/Users/数据集/ADHD200/Schaefer200"),
            "test":  Path(r"E:/Users/数据集/ADHD200/Schaefer200test"),
        }
        self.session_ids = ['session_1']
        self.input_dim = None
        self.atlas_type = "schaefer200"  # 或 "aal"
        self.model_dir = Path(r"E:/Users/mym910/PycharmProjects/论文复现/12/ABIDEII-SU_2/models")
