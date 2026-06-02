import os
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .encoders import ViT, TextTransformer
from .model import CLIP
from .loss import clip_loss
from .dataset import CLIPDataset
import tiktoken


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="path to pairs.txt (img_path<TAB>caption)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=2000, help="warmup steps")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--wandb", action="store_true", help="enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="clip-reproduction")
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


def load_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def build_model():
    vocab_size = tiktoken.get_encoding("r50k_base").n_vocab
    image_encoder = ViT(
        image_size=224, patch_size=32, in_channels=3,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0,
    )
    text_encoder = TextTransformer(
        vocab_size=vocab_size, context_length=77,
        embed_dim=512, depth=12, num_heads=8, mlp_ratio=4.0, dropout=0.0,
    )
    model = CLIP(image_encoder, text_encoder, embed_dim=512)
    return model


def train():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"clip-vit-b32-bs{args.batch_size}",
            config=vars(args),
        )

    pairs = load_pairs(args.data)
    print(f"Loaded {len(pairs)} image-text pairs")

    dataset = CLIPDataset(pairs, context_length=77)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                                  betas=(0.9, 0.98), eps=1e-6)

    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for images, tokens in pbar:
            images = images.to(device)
            tokens = tokens.to(device)

            # warmup
            if global_step < args.warmup:
                lr_scale = max(0.0, float(global_step) / max(1, args.warmup))
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            optimizer.zero_grad()

            ctx = torch.cuda.amp.autocast() if scaler else nullcontext()
            with ctx:
                img_feat, txt_feat, scale = model(images, tokens)
                loss = clip_loss(img_feat, txt_feat, scale)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_step += 1

            if global_step % args.log_every == 0:
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("logit_scale", scale.item(), global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
                if run:
                    run.log({
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "logit_scale": scale.item(),
                        "epoch": epoch,
                    }, step=global_step)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, os.path.join(args.save_dir, f"clip_epoch{epoch}.pt"))
        print(f"Saved checkpoint: epoch {epoch}")

    writer.close()
    if run:
        run.finish()


if __name__ == "__main__":
    train()
