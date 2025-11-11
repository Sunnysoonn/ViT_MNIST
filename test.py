import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import MNISTNet


def load_model(model_path='mnist_model.pth', device='cpu', 
               embed_dim=128, depth=4, num_heads=4):
    """加载训练好的模型"""
    model = MNISTNet(
        num_classes=10,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def test_model(model_path='mnist_model.pth', num_samples=10,
               embed_dim=128, depth=4, num_heads=4):
    """测试模型并显示预测结果"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载测试数据
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True
    )
    
    # 加载模型
    model = load_model(model_path, device, embed_dim, depth, num_heads).to(device)
    
    # 计算整体准确率
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f'整体测试准确率: {accuracy:.2f}%')
    print('-' * 60)
    
    # 可视化预测结果
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            data = data.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # 显示图像
            img = data.cpu().squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'真实: {target.item()}, 预测: {predicted.item()}\n'
                            f'置信度: {confidence.item()*100:.1f}%')
            axes[i].axis('off')
            
            # 打印预测信息
            if predicted.item() == target.item():
                status = '✓'
            else:
                status = '✗'
            print(f'样本 {i+1}: 真实标签={target.item()}, 预测标签={predicted.item()}, '
                  f'置信度={confidence.item()*100:.2f}% {status}')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    print(f'\n测试结果已保存到 test_results.png')
    plt.close()


def predict_single_image(model_path='mnist_model.pth', image_path=None,
                        embed_dim=128, depth=4, num_heads=4):
    """预测单张图片"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device, embed_dim, depth, num_heads).to(device)
    
    if image_path is None:
        # 如果没有提供图片路径，从测试集中随机选择一张
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        data, target = test_dataset[0]
        data = data.unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        print(f'预测结果: {predicted.item()}')
        print(f'置信度: {confidence.item()*100:.2f}%')
        print(f'真实标签: {target}')
        
        # 显示图片
        img = data.cpu().squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f'预测: {predicted.item()}, 真实: {target}')
        plt.axis('off')
        plt.show()
    else:
        # 这里可以添加从文件加载图片的代码
        print("从文件加载图片的功能需要额外实现")
        pass


if __name__ == '__main__':
    # 测试模型（参数需与训练时保持一致）
    test_model(
        model_path='mnist_model.pth', 
        num_samples=10,
        embed_dim=128,
        depth=4,
        num_heads=4
    )

