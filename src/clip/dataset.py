import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import tiktoken


class CLIPDataset(Dataset):
    def __init__(self, pairs, context_length=77):
        self.pairs = pairs
        self.context_length = context_length
        self.tokenizer = tiktoken.get_encoding("r50k_base")
        self.sos = self.tokenizer._special_tokens["<|endoftext|>"]
        self.eos = self.tokenizer._special_tokens["<|endoftext|>"]

        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __len__(self):
        return len(self.pairs)

    def tokenize(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.context_length - 2]
        tokens = [self.sos] + tokens + [self.eos]
        tokens = tokens + [0] * (self.context_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        tokens = self.tokenize(caption)
        return image, tokens
