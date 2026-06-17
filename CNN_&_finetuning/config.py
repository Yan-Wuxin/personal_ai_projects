import torch
import os

# 路径配置
DATA_DIR = "./dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_SAVE_PATH = "./cnn_model.pth"

# 超参数
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)  # 适配大多数经典预训练模型的输入尺寸
NUM_CLASSES = 10         # 10位主要核心女性角色
LR = 1e-4                # 微调时建议使用较小的学习率
EPOCHS = 30

# 硬件配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")