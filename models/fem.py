import torch.nn as nn

from .swin_transformer import PatchEmbed, PatchMerging, BasicLayer

'''
特征提取模块
feature extraction module, FEM
FEM由多个Swin提取块组成，分别从输入图像中提取特征。Swin提取块类似于Swin Transformer块。
该算法采用平移窗口和多头注意机制实现特征增强，并通过多个分块完成图像下采样。
FEM中提取的图像特征将被发送到MFM来完成特征融合和精细的图像预测

'''

class Down(nn.Module):
    def __init__(self, down_scale=2, in_dim=64, depths=(2, 2, 6, 2)):
        super(Down, self).__init__()
        self.inc = PatchEmbed(img_size=256, patch_size=down_scale, in_chans=6, embed_dim=in_dim,
                              norm_layer=nn.LayerNorm)
        # DownBlock 就是原论文中的 Swin Extraction Block：Swin Transformer Block + Patch Merging
        self.down1 = DownBlock(in_channels=in_dim, out_channels=in_dim * 2, resolution=256 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[0])
        self.down2 = DownBlock(in_channels=in_dim * 2, out_channels=in_dim * 4, resolution=128 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[1])
        self.down3 = DownBlock(in_channels=in_dim * 4, out_channels=in_dim * 8, resolution=64 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[2])
        self.down4 = DownBlock(in_channels=in_dim * 8, out_channels=in_dim * 8, resolution=32 // down_scale,
                               downsample=PatchMerging, cur_depth=depths[3])

    def forward(self, x):  # 残差结构
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

'''
DownBlock里面主要是由SwinTransformer Block + PatchMerging
PatchMerging 的作用是在经过嵌入+SwinTransformer后，假设每一个像素点都是一个块，这样把相同编码的块堆叠起来
假设一张图被分成四个大块，每一个块又被分成4个小块，按照1,2,3,4编码，那么PatchMerging合并就按照全是1的放一起，2的放一起堆叠，使得通道数变成原来的4倍

'''
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, downsample, cur_depth):
        super(DownBlock, self).__init__()
        self.layer = BasicLayer(dim=in_channels,
                                input_resolution=(resolution, resolution),
                                depth=cur_depth,
                                num_heads=in_channels // 32,
                                window_size=8,
                                mlp_ratio=1,
                                qkv_bias=True, qk_scale=None,
                                drop=0., attn_drop=0.,
                                drop_path=0.,
                                norm_layer=nn.LayerNorm)

        if downsample is not None:
            self.downsample = downsample((resolution, resolution), in_channels, out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        x_o = self.layer(x)

        if self.downsample is not None:
            x_o = self.downsample(x_o)

        return x_o

