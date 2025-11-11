# MNIST 手写数字分类项目

这是一个使用 PyTorch 实现的深度学习项目，使用 **Vision Transformer (ViT)** 架构对手写数字（0-9）进行分类。

## 项目结构

```
former/
├── model.py          # 模型定义（Vision Transformer网络）
├── train.py          # 训练脚本
├── test.py           # 测试脚本
├── requirements.txt  # 项目依赖
└── README.md         # 项目说明
```

## 环境要求

- Python 3.7+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

运行训练脚本开始训练模型：

```bash
python train.py
```

训练过程会：
- 自动下载 MNIST 数据集（如果未下载）
- 训练模型 10 个 epoch
- 显示训练损失和准确率
- 保存模型到 `mnist_model.pth`
- 生成训练曲线图 `training_curves.png`

### 2. 测试模型

运行测试脚本评估模型性能：

```bash
python test.py
```

测试脚本会：
- 加载训练好的模型
- 计算整体测试准确率
- 显示 10 个样本的预测结果
- 生成可视化结果图 `test_results.png`

## 模型架构

模型使用了 **Vision Transformer (ViT)** 架构：

### 核心组件：
1. **Patch Embedding**: 将 28×28 的图像分割成 7×7 的 patches（共 16 个patches）
2. **位置编码**: 为每个patch添加可学习的位置编码
3. **[CLS] Token**: 用于分类的特殊token
4. **Transformer Encoder**: 包含多个Transformer编码器块
   - 多头自注意力机制（Multi-Head Self-Attention）
   - 层归一化（Layer Normalization）
   - 前馈网络（MLP with GELU激活）
   - 残差连接
5. **分类头**: 使用[CLS] token的特征进行分类

### 默认参数：
- `embed_dim`: 128（embedding维度）
- `depth`: 4（Transformer层数）
- `num_heads`: 4（注意力头数）
- `patch_size`: 7（patch大小）
- `mlp_ratio`: 4.0（MLP扩展比例）

## 预期结果

经过 10 个 epoch 的训练后，Vision Transformer 模型在测试集上的准确率应该能够达到 **97%** 以上。

## 文件说明

- `model.py`: 定义了 `VisionTransformer` 类，实现了完整的 ViT 架构
  - `PatchEmbedding`: 图像patch嵌入
  - `MultiHeadSelfAttention`: 多头自注意力机制
  - `TransformerBlock`: Transformer编码器块
  - `VisionTransformer`: 完整的ViT模型
- `train.py`: 训练脚本，包含数据加载、训练循环和模型保存
- `test.py`: 测试脚本，包含模型评估和结果可视化
- `mnist_model.pth`: 训练好的模型权重（训练后生成）
- `training_curves.png`: 训练过程的可视化曲线（训练后生成）
- `test_results.png`: 测试结果的可视化（测试后生成）

## 自定义参数

可以在 `train.py` 中修改以下参数：

### 训练参数：
- `num_epochs`: 训练轮数（默认：10）
- `batch_size`: 批次大小（默认：64）
- `learning_rate`: 学习率（默认：0.001）

### 模型参数：
- `embed_dim`: embedding维度（默认：128）
- `depth`: Transformer层数（默认：4）
- `num_heads`: 注意力头数（默认：4，必须能被embed_dim整除）
- `patch_size`: patch大小（默认：7）
- `mlp_ratio`: MLP扩展比例（默认：4.0）

**注意**: 测试时使用的模型参数必须与训练时保持一致！

## 注意事项

- 首次运行会自动下载 MNIST 数据集（约 60MB）
- 如果使用 GPU，训练速度会显著提升
- 模型文件 `mnist_model.pth` 会在训练后自动生成
- Transformer模型相比CNN需要更多的训练时间，但通常能达到相似的准确率
- 模型参数数量会根据 `embed_dim`、`depth` 和 `num_heads` 的变化而改变
- 使用 AdamW 优化器，并添加了权重衰减（weight_decay=0.01）以防止过拟合

## 常见问题

### 1. SSL证书验证错误

如果遇到 SSL 证书验证错误（`SSL: CERTIFICATE_VERIFY_FAILED`），代码已自动处理。如果仍然失败，可以尝试：

```bash
# 更新certifi包
pip install --upgrade certifi

# 或者在Python中设置环境变量
export PYTHONHTTPSVERIFY=0  # macOS/Linux
set PYTHONHTTPSVERIFY=0     # Windows
```

### 2. 下载失败

如果自动下载失败，可以：

1. **检查网络连接**：确保可以访问互联网
2. **手动下载数据集**：
   - 访问 [MNIST数据集下载页面](http://yann.lecun.com/exdb/mnist/)
   - 下载以下文件到 `./data/MNIST/raw/` 目录：
     - `train-images-idx3-ubyte.gz`
     - `train-labels-idx1-ubyte.gz`
     - `t10k-images-idx3-ubyte.gz`
     - `t10k-labels-idx1-ubyte.gz`
   - 重新运行训练脚本，它会自动解压和处理

3. **使用镜像源**：代码已自动处理SSL问题，如果仍有问题，可以尝试配置代理

### 3. 数据集已存在

如果数据集已经下载，脚本会自动检测并跳过下载步骤。

## Transformer vs CNN

本项目使用 Vision Transformer 架构，相比传统的CNN：
- **优势**: 
  - 全局感受野，能够捕获长距离依赖
  - 自注意力机制可以学习图像不同部分之间的关系
  - 架构更统一，易于扩展
- **特点**: 
  - 需要更多的训练数据才能发挥优势（MNIST数据量较小）
  - 训练时间可能略长于简单CNN
  - 在MNIST这样的小数据集上，CNN和Transformer的表现相近

