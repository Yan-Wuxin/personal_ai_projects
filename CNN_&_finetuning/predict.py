import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import config
from dataset import get_data_transforms, get_dataloaders
from model import get_character_classifier

@torch.no_grad()
def predict_single_image(image_path, model, classes, transform, device): # 预测单张输入图片的类别
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    class_idx = predicted.item()
    return classes[class_idx]

@torch.no_grad()
def visualize_test_batch(model, dataloader, classes, device, num_images=8): # 批量可视化推理测试Batch中的前几张图片（对应原Jupyter中的特定需求）
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    _, preds = outputs.max(1)
    
    plt.figure(figsize=(15, 8))
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(2, 4, i + 1)
        # 反归一化以便正常显示彩色图像
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        import numpy as np
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        title_color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}", color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载标签映射和转换器
    _, valid_loader, classes = get_dataloaders()
    _, valid_transform = get_data_transforms()
    
    # 实例化并载入已训练好的权重
    model = get_character_classifier()
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        print("开始执行批量测试可视化...")
        visualize_test_batch(model, valid_loader, classes, config.DEVICE)
    else:
        print(f"未找到训练好的权重文件：{config.MODEL_SAVE_PATH}，请先运行 train.py 训练模型。")

if __name__ == "__main__":
    main()