import os
import json
import shutil
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
from dataset import get_dataloaders
from model import MyCNNModel


def inference_and_export_json():
    _, _, test_loader, _, _, _ = get_dataloaders()
    model = MyCNNModel(num_classes=len(config.LABEL2CLASS)).to(config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(config.SAVE_DIR, 'best_model.pth'), map_location=config.DEVICE),
                          strict=False)
    model.eval()

    results = []
    idx_list = []

    print("Running inference and generating JSON...")
    with torch.no_grad():
        for images, idx in tqdm(test_loader):
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            results.extend(preds)
            idx_list.extend(idx.cpu().tolist())

    STUDENT_ID = 'ave_classification'
    final_output = []
    for (img_id, result) in zip(idx_list, results):
        out = {
            "id": img_id,
            "pred": result
        }
        final_output.append(out)

    output_path = os.path.join(config.SAVE_DIR, f'{STUDENT_ID.lower()}.json')
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"JSON prediction saved to {output_path}")


def generate_inference_images(): # 将推理结果按照类别名字分类并导出为图片放入子文件夹中
    _, _, test_loader, _, _, _ = get_dataloaders()

    if os.path.exists(config.OUTPUT_DIR):
        shutil.rmtree(config.OUTPUT_DIR)
    for name in config.LABEL2CLASS.keys():
        os.makedirs(os.path.join(config.OUTPUT_DIR, name), exist_ok=True)

    num_to_class_map = {v: k for k, v in config.LABEL2CLASS.items()}

    model = MyCNNModel(num_classes=len(config.LABEL2CLASS)).to(config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(config.SAVE_DIR, 'best_model.pth'), map_location=config.DEVICE),
                          strict=False)
    model.eval()

    print("Generating classified images into folders...")
    with torch.no_grad():
        for images, idx in tqdm(test_loader):
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()

            for i in range(len(preds)):
                pred_idx = preds[i]
                pred_label = num_to_class_map[pred_idx]
                img_id = idx[i].item() if torch.is_tensor(idx[i]) else idx[i]

                img_tensor = images[i].cpu()
                img_save_path = os.path.join(config.OUTPUT_DIR, pred_label, f"{img_id}.png")
                save_image(img_tensor, img_save_path)
    print(f"Classified images saved successfully to {config.OUTPUT_DIR}")


def visualize_test_batch(n_images=6): # 可视化单个Batch测试集图片
    _, _, test_loader, _, _, _ = get_dataloaders()
    images, idx = next(iter(test_loader))

    if len(images) > 6 + n_images:
        images = images[6:6 + n_images]
        idx = idx[6:6 + n_images]
    else:
        images = images[:n_images]
        idx = idx[:n_images]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(n_images, len(images))):
        img = images[i]
        img = torch.clamp(img, 0, 1)
        img_np = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(f"ID: {idx[i].item() if torch.is_tensor(idx[i]) else idx[i]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inference_and_export_json()
    generate_inference_images()