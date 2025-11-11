import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """将图像分割成patches并转换为embedding"""
    
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=128):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层将patch转换为embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.proj(x)  # [B, embed_dim, 4, 4]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        scale = (self.head_dim ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 残差连接 + 层归一化 + 注意力
        x = x + self.attn(self.norm1(x))
        # 残差连接 + 层归一化 + MLP
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer模型用于MNIST手写数字分类"""
    
    def __init__(self, img_size=28, patch_size=7, in_channels=1, 
                 num_classes=10, embed_dim=128, depth=4, num_heads=4, 
                 mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化位置编码
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        # 使用正态分布初始化位置编码和CLS token（这些是可学习参数）
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # x: [B, 1, 28, 28]
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, n_patches+1, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)
        
        # 层归一化
        x = self.norm(x)
        
        # 使用[CLS] token进行分类
        cls_token_final = x[:, 0]
        
        # 分类头
        logits = self.head(cls_token_final)
        
        return logits


# 为了保持兼容性，保留MNISTNet作为别名
MNISTNet = VisionTransformer

