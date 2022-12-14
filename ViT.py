import einops
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    # multihead attention if mask !=None
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)  # b n 3*dim
        # to_qkv 通过 fc 将x dim 扩大3 倍，变成三个矩阵 q，k，v
        q, k, v = einops.rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # 3 b h n dim/h
        # 矩阵乘
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # b h n n
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1]
            mask = mask[:, None, :] * mask[:, :, None]
            # 查！
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # b h n dim/h
        out = einops.rearrange(out, 'b h n d -> b n (h d)')  # b n dim
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, depth, dim, heads, mlp_dim, channels=3):
        '''
        :param img_size: 图片尺寸
        :param patch_size: 每个patch大小，正方形，因此img_size要能够整除patch_size。
        :param num_classes:
        :param depth: transfromer深度，Transformer Encoder网络重复次数，网络深度。
        :param dim: 网络参数，也是多头注意力的特征长度，因此dim要能够整除dim。
        :param heads: 多头注意力个数。
        :param mlp_dim: 全连接分类器隐含层参数维度。
        :param channels:
        '''
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embeding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        # 图片切成小图片的大小
        p = self.patch_size
        # 图片维度变换 n c h w->b h*w/(p*p) p*p*c
        x = einops.rearrange(img, 'b c (h p1) (w p2)  -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # 通过个 fc 将tensor 维度： b h*w/(p*p) patchdim -> b h*w/(p*p) dim
        x = self.patch_to_embedding(x)
        # 扩展成  b 1 dim
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        # b h*w/(p*p)+1 dim
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embeding
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
