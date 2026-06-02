import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patchs = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, img_size, img_size)
        x = self.proj(x) # (B, embed_dim, num_patchs, num_patchs)
        x = x.flatten(2) # (B, embed_dim, N)
        x = x.transpose(1, 2) # (B, N, embed_dim)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forwward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x) # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        x = attn @ v # (B, num_heads, N, head_dim)
        x.transpose(1, 2) # (B, N, num_heads, head_dim)
        x.reshape(B, N, D)

        x = self.proj(x) # (B, N, D)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.layer(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patchs = self.patch_embed.num_patchs

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patchs + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, N, D)
        
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1) # (B, N+1, D)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0] # (B, D) 仅做特征提取
    
class TextTransformer(nn.Module):
    def __init__(self, vocab_size=49408, context_length=77, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, embed_dim) # 对每个 token，都把它映射成 embed_dim
        self.pos_embed = nn.Parameters(torch.zeros(1, context_length, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward(self, x):
        B, L = x.shape # 这里 L=context_length
        x = self.token_embed(x) # (B, L, embed_dim)
        x = x + self.pos_embed # (B, L, embed_dim)，广播
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x