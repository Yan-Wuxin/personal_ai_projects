import numpy as np
import os
from utils.data_loader import load_data
from models.L_layer_model import L_layer_model

def train_and_save_model():
    # 训练模型并保存参数
    
    print("=== 开始训练模型 ===")
    
    # 加载数据
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # 数据预处理
    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    num_px = train_x_orig.shape[1]

    train_x_flatten = train_x_orig.reshape(m_train, -1).T
    test_x_flatten = test_x_orig.reshape(m_test, -1).T

    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("训练集样本数：" + str(m_train))
    print("测试集样本数：" + str(m_test))
    print("图像大小：(" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    # 定义网络结构
    layers_dims = [12288, 20, 7, 6, 1]

    # 训练模型
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=3000, print_cost=True)

    # 保存模型参数
    os.makedirs('trained_models', exist_ok=True)
    np.save('trained_models/parameters.npy', parameters)
    print("=== 模型训练完成并已保存 ===")
    
    return parameters, classes

if __name__ == "__main__":
    train_and_save_model()