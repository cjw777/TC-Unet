from functools import partial

import torch
from timm.models.layers import DropPath
from torch import nn
import torch.nn.functional as F
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),trans_down=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,out_features = dim, act_layer=act_layer, drop=drop)

        self.down_to = nn.Linear(dim,dim*2,bias = False)
        self.pool = nn.MaxPool2d(2,2)

        self.trans_down = trans_down
        if trans_down:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim*2))

    def forward(self, x):

        #block
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))



        if self.trans_down:
            # 下采样,用于下阶段
            xd = x[:, 1:, :]
            b, n, c = xd.shape
            xd = xd.permute(0, 2, 1).view(b, c, int(np.sqrt(n)), int(np.sqrt(n))).contiguous()
            xd = self.pool(xd)
            xd = xd.view(b, c, n // 4).permute(0, 2, 1)
            xd = self.down_to(xd)
            #cls_token
            cls_tokens = self.cls_token.expand(b, -1, -1)
            xd = torch.cat([cls_tokens,xd],1)
            return x, xd
        else:
            return x,x






class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes,act_layer = nn.GELU,
                 norm_layer = partial(nn.LayerNorm, eps = 1e-6)):
        super(FCUDown, self).__init__()


        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size = 1, stride = 1, padding = 0)
        self.sample_pooling = nn.AvgPool2d(kernel_size = 4, stride = 4)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()



    def forward(self, x, x_t):
        x = self.conv_project(x)  # torch.Size([1, 768, 56, 56])

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)  # torch.Size([1, 196, 768])
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim = 1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes,act_layer = nn.ReLU,
                 norm_layer = partial(nn.BatchNorm2d, eps = 1e-6)):
        super(FCUUp, self).__init__()


        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size = 1, stride = 1, padding = 0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        B, HW, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, scale_factor = 4,mode = 'bilinear',align_corners = True)