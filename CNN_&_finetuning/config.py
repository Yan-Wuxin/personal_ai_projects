import torch

# 随机种子与设备配置
SEED = 23
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集元数据及图片目录路径
TRAIN_JSON_PATH = './train/train.json'
TRAIN_IMG_DIR = './train'
VALID_JSON_PATH = './valid/valid.json'
VALID_IMG_DIR = './valid'
TEST_JSON_PATH = './test/test.json'
TEST_IMG_DIR = './test'

# 模型保存与推理输出路径
SAVE_DIR = "./linear/"
OUTPUT_DIR = './inference_result'

# 模型训练超参数
BATCH_SIZE = 64
NUM_EPOCHS = 80
LR = 3e-5
WEIGHT_DECAY = 1e-4

# 标签与类别映射
LABEL2CLASS = {
    "anon": 0,
    "mutsumi": 1,
    "nyamuchi": 2,
    "rana": 3,
    "sakiko": 4,
    "soyo": 5,
    "taki": 6,
    "tomori": 7,
    "uika": 8,
    "umiri": 9
    ## "mana": 10   ...only appears in few sences...
}