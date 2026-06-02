import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, logit_scale):
    logits = image_features @ text_features.T * logit_scale # (B, B)

    B = logits.shape
    labels = torch.arange(B, device=logits.device)

    loss_i = F.cross_entropy(logits, labels) # 对每一行做 softmax，然后计算 -log(row[i])，求平均，目的是让对角线的概率尽可能大
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2 