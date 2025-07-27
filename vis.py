import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_segmentation_results(model_dir='data/models', num_samples=5, random_seed=None):
    """
    可视化分割模型的预测结果
    
    参数:
    model_dir: 包含.npy文件的目录路径
    num_samples: 要显示的样本数
    random_seed: 随机数生成器的种子，用于结果可重现
    """
    model_dir = Path(model_dir)
    
    # 加载数据
    images = np.load(model_dir / 'segmenter_images.npy')
    preds = np.load(model_dir / 'segmenter_preds.npy')
    true = np.load(model_dir / 'segmenter_true.npy')
    
    # 确保每次运行时随机选择相同的样本（如果指定了种子）
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 随机选择样本索引
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    # 创建一个图形
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    
    # 如果只有一个样本，确保axes是二维数组
    if len(indices) == 1:
        axes = np.array([axes])
    
    # 遍历每个样本
    for i, idx in enumerate(indices):
        # 显示原始图像（假设图像已经被归一化到[0,1]）
        # img = images[idx].transpose(1, 2, 0)  # 调整通道顺序：[C,H,W] -> [H,W,C]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        # 先反归一化：(x * std) + mean
        img_denorm = (images[idx] * std[:, np.newaxis, np.newaxis]) + mean[:, np.newaxis, np.newaxis]  
        # 再转通道顺序为 [H,W,C] 并缩放到 [0,255]
        img = np.clip(img_denorm.transpose(1,2,0) * 255, 0, 255).astype(np.uint8)  
        # print(img)
        
        # 如果图像被归一化，尝试恢复到[0,255]范围
        if img.max() <= 1.0:
            img = img * 255
        img = img.astype(np.uint8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('RGB')
        axes[i, 0].axis('off')
        
        # 显示预测掩码
        pred_mask = preds[idx]
        axes[i, 1].imshow(pred_mask, cmap='viridis')
        axes[i, 1].set_title('Preditcted')
        axes[i, 1].axis('off')
        
        # 显示真实掩码
        true_mask = true[idx]
        axes[i, 2].imshow(true_mask, cmap='viridis')
        axes[i, 2].set_title('Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(model_dir / 'segmentation_results.png')
    print(f"可视化结果已保存到: {model_dir / 'segmentation_results.png'}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--model_dir', type=str, default='data/models', help='Output Directory')
    parser.add_argument('--num_samples', type=int, default=1, help='Samples')
    parser.add_argument('--seed', type=int, default=None, help='Seed')
    args = parser.parse_args()
    
    # 检查文件是否存在
    model_dir = Path(args.model_dir)
    required_files = ['segmenter_images.npy', 'segmenter_preds.npy', 'segmenter_true.npy']
    for file in required_files:
        if not (model_dir / file).exists():
            print(f"错误: 文件 {model_dir / file} 不存在")
            return
    
    visualize_segmentation_results(
        model_dir=args.model_dir,
        num_samples=args.num_samples,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()