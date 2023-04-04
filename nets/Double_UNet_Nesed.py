from functools import partial

import torch
from torch import nn
from nets.attention import ChannelAttention
from nets.attention import SpatialAttention
from nets.UNet_Nested import *
import torch.nn.functional as F
from nets.trans_utils import *


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, middle_channels, outplanes, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., trans_down=True):

        super(ConvTransBlock, self).__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, outplanes))
        # self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)
        self.cnn_block = VGGBlock(inplanes,middle_channels,outplanes)

        self.fcu_down = FCUDown(outplanes, outplanes)

        self.trans_block = Block(dim = outplanes, num_heads = num_heads, mlp_ratio = mlp_ratio,
                                 qkv_bias = qkv_bias, qk_scale = qk_scale, drop = drop_rate,
                                 attn_drop = attn_drop_rate, drop_path = drop_path_rate, trans_down=trans_down)

        self.fcu_up = FCUUp(inplanes=outplanes, outplanes= outplanes)

        self.fusion_block = VGGBlock(outplanes, middle_channels, outplanes)

        self.trans_down = trans_down

    def forward(self, x, x_t):
        #x2 用于trans分支
        x= self.cnn_block(x)#torch.Size([1, 256, 56, 56]) torch.Size([1, 64, 56, 56])

        _, _, H, W = x.shape

        #FCUDown 融合
        x_st = self.fcu_down(x, x_t)#torch.Size([1, 197, 768])

        #transblock
        x_t_,x_t = self.trans_block(x_st + x_t)#torch.Size([1, 197, 768])

        #FCUUp
        x_t_r = self.fcu_up(x_t_)#torch.Size([1, 64, 56, 56])

        #cnn 分支融合
        x = self.fusion_block(x+x_t_r)#torch.Size([1, 256, 56, 56])

        if self.trans_down:
            return x, x_t
        else:
            return x,x_t_r

class Double_UNetNesed(nn.Module):
    def __init__(self, num_classes, input_channels = 3, deep_supervision = False,**kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.cls_token = nn.Parameter(torch.zeros(1, 1, nb_filter[0]))

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels,nb_filter[0],kernel_size = 7,stride = 2,padding = 3),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)

        self.conv0_0 = ConvTransBlock(input_channels, nb_filter[0], nb_filter[0],**kwargs)
        # self.cot0 = CT(nb_filter[0], nb_filter[0])
        self.conv1_0 = ConvTransBlock(nb_filter[0], nb_filter[1], nb_filter[1],**kwargs)
        self.conv2_0 = ConvTransBlock(nb_filter[1], nb_filter[2], nb_filter[2],**kwargs)
        # self.cot1 = CT(nb_filter[2], nb_filter[2])
        self.conv3_0 = ConvTransBlock(nb_filter[2], nb_filter[3], nb_filter[3],**kwargs,)
        # self.cot2 = CT(nb_filter[3], nb_filter[3])
        self.conv4_0 = ConvTransBlock(nb_filter[3], nb_filter[4], nb_filter[4],**kwargs)
        # self.cot3 = CT(nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        # self.conv3_1 = ASPP(nb_filter[3] + nb_filter[4], nb_filter[3], rate=[3, 6, 9])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        # self.conv2_2 = ASPP(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], rate=[2, 4, 6])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        # self.conv1_3 = ASPP(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], rate=[2, 4, 6])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)

    def forward(self, input):
        B,_,_,_ = input.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_stem = self.pool(self.stem(input)).flatten(2).transpose(1, 2)
        x_stem = torch.cat([cls_tokens, x_stem], dim = 1)


        x0_0,t0_0 = self.conv0_0(input,x_stem)
        # x0_0 = self.cot0(x0_0)
        x1_0,t1_0= self.conv1_0(self.pool(x0_0),t0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print(x0_0.shape)

        x2_0,t2_0 = self.conv2_0(self.pool(x1_0),t1_0)
        # x2_0 = self.cot1(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print(x0_2.shape)

        x3_0,t3_0 = self.conv3_0(self.pool(x2_0),t2_0)
        # x3_0 = self.cot2(x3_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print(x0_3.shape)

        #与x3_0相加融合进入aspp
        x4_0,t4_0 = self.conv4_0(self.pool(x3_0),t3_0)
        # x4_0 = self.cot3(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print(x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output