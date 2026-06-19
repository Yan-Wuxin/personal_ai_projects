import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
from dataset import get_dataloaders
from model import MyCNNModel


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model():
    setup_seed(config.SEED)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    train_loader, val_loader, _, train_set, val_set, _ = get_dataloaders()
    model = MyCNNModel(num_classes=len(config.LABEL2CLASS)).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_acc = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0

        # 训练
        print("=" * 20 + f"Epoch {epoch + 1}" + "=" * 20)
        for images, labels, idx in tqdm(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_set)
        print(f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2%}")

        # 验证
        val_correct = 0
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels, idx in tqdm(val_loader):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / len(val_set)
        print(f"Loss={avg_val_loss:.4f}, Acc={val_acc:.2%}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print("当前最佳模型参数已保存")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    # 绘制并保存曲线图
    epochs = range(1, config.NUM_EPOCHS + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Validation Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, 'training_curve.png'))
    print(f"训练完成，模型已保存至{config.SAVE_DIR}")


if __name__ == "__main__":
    train_model()