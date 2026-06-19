# BanG Dream! Character Image Classifier (CNN & Finetuning)

本项目基于国立台湾大学李宏毅教授课程（GenAI-ML Fall 2025）的 Homework 6 实现。通过利用经典的深度学习与计算机视觉技术，构建了一个高质量的图像分类器，专门用于精准识别和区分知名IP《BanG Dream! It's MyGO!!!!!》以及《BanG Dream! Ave Mujica》中的 10 位核心女性角色

## 核心任务与TODO目标
* **深度学习框架**：完全基于 PyTorch 框架搭建底层模型的训练与推理架构
* **CNN 架构**：从零构建卷积神经网络CNN，用于高维图像特征提取
* **数据增强 (Data Augmentation)**：实现并组合了多种数据增强策略，用于大幅提升模型的泛化能力与抗过拟合性
* **迁移学习 & 微调**：引入并重构了预定义的高级图像网络（ ResNet ）并载入预训练权重进行高效微调（Fine-tuning）
* **特别优化**：重点优化并使其有能力区分容易混淆的角色

## 数据增强策略 (Data Augmentation)
为了解决样本不平衡和过拟合问题，本项目在训练流水线中应用了多维数据增强：
- 随机水平翻转 (`RandomHorizontalFlip`)
- 随机角度旋转 (`RandomRotation`)
- 颜色抖动调整 (`ColorJitter`)
- 尺寸裁剪与标准化 (`Resize & Normalize`)

## 实验表现与训练可视化
- **计算设备**： Tesla T4 GPU 15GB（Google colab）
- **训练表现**：
  - **训练集 (Train Set)**：模型在经过 60+ Epoch 的迭代后，训练集准确率逐渐逼近100.00，交叉熵损失（Cross-Entropy Loss）收敛至约 0.0166
  - **验证集 (Validation Set)**：验证集上的分类准确率达到 90% 以上
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/a119a31d-5f82-43fe-92dc-0fbfdd5b0d44" />

## 文件目录结构
```text
├── GenAI-ML-HW6.ipynb      # 原项目文件，包含数据导入、增强、模型微调与评估的核心Colab笔记本
│
├── config.py          # 1. 配置文件（存放超参数、路径等）
├── dataset.py         # 2. 数据处理与数据增强（Dataset & DataLoader）
├── model.py           # 3. 模型架构定义（CNN / 迁移学习微调）
├── train.py           # 4. 训练与验证主程序
├── predict.py         # 5. 单张/批量图像推理与可视化的预测脚本
│
├── ave_classification.json # 测试结果（json格式）
├── AVE_Test_Results.zip # 测试结果（分类结果）
├── README.md          # 项目说明文档
└── .gitignore         # 忽略数据和权重文件
