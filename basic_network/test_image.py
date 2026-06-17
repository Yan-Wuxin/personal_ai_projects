import numpy as np
import os
from utils.data_loader import load_data
from utils.predictors import predict

def load_trained_model():
    # 加载已训练的模型参数
    model_path = 'trained_models/parameters.npy'
    if not os.path.exists(model_path):
        print("错误: 未找到训练好的模型，请先运行训练新模型")
        return None
    
    parameters = np.load(model_path, allow_pickle=True).item()
    print("模型加载成功")
    return parameters

def test_single_image(image_path, true_label, parameters, classes):
    # 测试单张图片
    # 加载数据获取图片尺寸信息
    _, _, _, _, _ = load_data()
    num_px = 64  # 根据数据集确定
    
    try:
        # 尝试导入图片处理相关库
        from PIL import Image
        
        # 加载和预处理图片
        image = np.array(Image.open(image_path))
        
        # 调整图片尺寸
        img = Image.fromarray(image)
        img = img.resize((num_px, num_px), Image.LANCZOS)
        my_image = np.array(img).reshape((num_px * num_px * 3, 1))
        
        # 进行预测
        my_predicted_image = predict(my_image, true_label, parameters)
        
        # 输出预测结果
        prediction = int(np.squeeze(my_predicted_image))
        class_name = classes[prediction].decode("utf-8")
        
        print(f"图片: {os.path.basename(image_path)}")
        print(f"预测结果: y = {prediction}")
        print(f"模型预测为: '{class_name}' 图片")
        
        # 环境支持时显示图片
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方块的问题
            plt.imshow(image)
            plt.title(f"预测: {class_name} (y={prediction})")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"注意: 无法显示图片 ({e})，但预测已完成")
            
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装 Pillow 和 matplotlib")
    except Exception as e:
        print(f"处理图片时出错: {e}")

def test_multiple_images(parameters, classes):
    # 测试多张图片
    images_dir = 'images'
    if not os.path.exists(images_dir):
        print(f"错误: 图片目录 '{images_dir}' 不存在")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"在 '{images_dir}' 目录中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张测试图片:")
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        print("\n" + "="*50)
        test_single_image(image_path, [0], parameters, classes)

if __name__ == "__main__":
    # 加载模型
    parameters = load_trained_model()
    if parameters is None:
        exit(1)
    
    # 加载类别信息
    _, _, _, _, classes = load_data()
    
    # 测试所有图片
    test_multiple_images(parameters, classes)