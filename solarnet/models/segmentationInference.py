import torch
from torch import nn
from .base import ResnetBase
from solarnet.models import Segmenter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SegmentationInference():
    """图像分割推理类，提供静态方法进行模型推理"""
    
    def preprocess_image(image_path, size=(224, 224)):
        """加载并预处理图像"""
        MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
        img_array = np.array(img) / 255.0  # 将像素值缩放到 [0,1]
        
        # 减去均值并除以标准差
        img_array = (img_array - MEAN) / STD
        
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 调整维度为 [B, C, H, W]
        return img_tensor
    
    def load_model(model, model_path, device='cpu', imagenet_base=False):
        """加载预训练模型"""
        print(f"DEBUG: model_path={model_path}, device={device}, imagenet_base={imagenet_base}")

        model = Segmenter(imagenet_base=imagenet_base)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    
    def postprocess_output(output, threshold=0.5):
        """处理模型输出"""
        mask = output.cpu().squeeze().numpy()
        # binary_mask = (mask > threshold).astype(np.uint8)
        return mask
    
    def visualize_results(original_image, segmentation_mask):
        """可视化原始图像和分割结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(segmentation_mask, cmap='viridis')
        ax2.set_title('Segmentation Mask')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def setp_forward(the, model, image_path, output_path=None, threshold=0.5, visualize=False):
        """
        模型推理主函数
        
        参数:
            model: 加载好的预训练模型
            image_path: 输入图像路径
            output_path: 输出掩码保存路径(可选)
            threshold: 二值化阈值，默认为0.5
            visualize: 是否可视化结果，默认为False
        
        返回:
            分割掩码的numpy数组
        """
        device = next(model.parameters()).device
        
        # 预处理图像
        input_tensor = SegmentationInference.preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            output = model(input_tensor)
        
        # 后处理
        mask = SegmentationInference.postprocess_output(output, threshold)
        
        # 保存结果
        if output_path:
            np.save(output_path, mask)
        
        # 可视化
        if visualize:
            original_image = Image.open(image_path).convert('RGB')
            SegmentationInference.visualize_results(original_image, mask)
        
        # 清理钩子
        if hasattr(model, 'cleanup'):
            model.cleanup()
            
        return mask
