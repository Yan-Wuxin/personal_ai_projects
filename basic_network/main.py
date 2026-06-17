import os
import sys

def main():
    print("简易深度学习猫分类器")
    
    while 1:
        print("=" * 30)
        print("1. 训练新模型")
        print("2. 测试图片")
        print("3. 退出")

        choice = input("请选择操作 (1/2/3): ").strip()
        
        if choice == "1":
            # 训练模型
            from train_model import train_and_save_model
            train_and_save_model()
            
        elif choice == "2":
            # 测试图片
            from test_image import load_trained_model, test_multiple_images
            from utils.data_loader import load_data
            
            parameters = load_trained_model()
            if parameters is not None:
                _, _, _, _, classes = load_data()
                test_multiple_images(parameters, classes)
                
        elif choice == "3":
            print("再见!")
            return
            
        else:
            print("无效选择")

if __name__ == "__main__":
    main()