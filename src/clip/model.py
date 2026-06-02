import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 投影头，把图像/文本投影到同一多模态空间
        self.image_proj = nn.Linear(self.image_encoder.embed_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(self.text_encoder.embed_dim, embed_dim, bias=False)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # 用来调节 softmax 的强弱，如果 scale 大，表示 softmax 分布更尖锐

    def encode_image(self, image):
        x = self.image_encoder(image) # (B, D_image)
        x = self.image_proj(x) # (B, embed_dim)
        return F.normalize(x, dim=-1) # 方便计算余弦相似度，不用除以模长
    
    def encode_text(self, text):
        x = self.text_encoder(text)
        x = self.text_proj(x)
        return F.normalize(x, dim=-1)
    
    def forward(self, image, text):
        img_feat = self.encode_image(image)
        txt_feat = self.encode_text(text)
        scale = self.logit_scale.exp() # 温度系数
        return img_feat, txt_feat, scale