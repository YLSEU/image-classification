# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):  # [512, 65, 512]
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # [512, 65, 512]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # [512, 8, 65, 64]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [512, 8, 65, 65]

        attn = self.attend(dots)  # [512, 8, 65, 65]

        out = torch.matmul(attn, v)  # # [512, 8, 65, 65] * # [512, 8, 65, 64] = [512, 8, 65, 64]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [512, 8, 65, 64] ---> # [512, 65, 8 * 64]

        return self.to_out(out)  # [512, 65, 512] ---> [512, 65, 512]

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [512, 3, 32, 32] ---> [512, 64, 48] ---> [512, 64, 512]  64 patches
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # [1, 1, 512] ---> [512, 1, 512]
        x = torch.cat((cls_tokens, x), dim=1)  # [512, 65, 512]
        x += self.pos_embedding[:, :(n + 1)]  # [512, 65, 512]
        x = self.dropout(x)

        x = self.transformer(x)  # [512, 65, 512] ---> [512, 65, 512]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # [512, 512] 只取第一个patches 即tokens

        x = self.to_latent(x)

        return self.mlp_head(x)


def ViT_Base(img_size, num_class):
    net = ViT(
            image_size=img_size,
            patch_size=16,
            num_classes=num_class,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1
        )
    
    return net


def ViT_Large(img_size, num_class):
    net = ViT(
            image_size=img_size,
            patch_size=16,
            num_classes=num_class,
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=4096,
            dropout=0.1,
            emb_dropout=0.1
        )
    
    return net


def ViT_Huge(img_size, num_class):
    net = ViT(
            image_size=img_size,
            patch_size=14,
            num_classes=num_class,
            dim=1280,
            depth=32,
            heads=16,
            mlp_dim=5120,
            dropout=0.1,
            emb_dropout=0.1
        )
    
    return net


if __name__ == '__main__':
    model = ViT_Huge(img_size=224, num_class=1000)
    print(model)
