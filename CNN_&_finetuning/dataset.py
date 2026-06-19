import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config


class MyGOCharacterDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        # 加载元数据
        with open(json_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        attr = self.metadata[idx]
        img_path = os.path.join(self.images_dir, attr['filename'])
        img_id = attr["id"]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        if "test" in self.images_dir: # 测试集无标签
            return image, img_id
        else:
            label = attr['label']
            label = config.LABEL2CLASS[label]
            return image, label, img_id


def get_transforms(): # 考虑到发色和瞳孔颜色对二次元形象识别的重要性，故不进行随机灰度变换
    train_transform = transforms.Compose([
        transforms.Resize((144, 144)),  # 轻微放大
        transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomCrop((128, 128)),  # 随机裁剪但维持原图片尺寸
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([ # 测试集不做数据增强操作
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return train_transform, test_transform


def get_dataloaders():
    train_transform, test_transform = get_transforms()

    train_set = MyGOCharacterDataset(json_path=config.TRAIN_JSON_PATH, images_dir=config.TRAIN_IMG_DIR,
                                     transform=train_transform)
    val_set = MyGOCharacterDataset(json_path=config.VALID_JSON_PATH, images_dir=config.VALID_IMG_DIR,
                                   transform=train_transform)
    test_set = MyGOCharacterDataset(json_path=config.TEST_JSON_PATH, images_dir=config.TEST_IMG_DIR,
                                    transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_set, val_set, test_set