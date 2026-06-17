import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import config

def get_data_transforms(): # 获取训练和验证/测试集的数据转换流程
    # 强力数据增强组合
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),          # 随机水平翻转
        transforms.RandomRotation(degrees=15),           # 随机角度旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet标准归一化
    ])
    
    # 验证/测试集保持原样，仅做缩放和归一化
    valid_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, valid_transform

def get_dataloaders(): # 构建训练和验证集的 DataLoader
    train_transform, valid_transform = get_data_transforms()
    
    # 自动识别子文件夹作为类别标签
    train_dataset = ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    valid_dataset = ImageFolder(root=config.VALID_DIR, transform=valid_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(valid_dataset)}")
    print(f"类别映射关系: {train_dataset.class_to_idx}")
    
    return train_loader, valid_loader, train_dataset.classes