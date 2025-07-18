from __future__ import absolute_import

import torch.nn as nn

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import torch


__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3','MobileNetV3','AlexNetV2_MixAttn_DW']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


# class _AlexNet(nn.Module):
#
#     def forward(self, x):
#
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         return x

class _AlexNet(nn.Module):
    def forward(self, x):
        x = self.conv1(x)
        if hasattr(self, "attn1"):
            x = self.attn1(x)

        x = self.conv2(x)
        if hasattr(self, "attn2"):
            x = self.attn2(x)

        x = self.conv3(x)
        if hasattr(self, "attn3"):
            x = self.attn3(x)

        x = self.conv4(x)
        if hasattr(self, "attn4"):
            x = self.attn4(x)

        x = self.conv5(x)
        if hasattr(self, "attn5"):
            x = self.attn5(x)

        return x



class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))

####################注意力模块##################
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel Attention
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        channel_att = self.mlp(avg) + self.mlp(max)
        channel_att = self.sigmoid_channel(channel_att).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att))
        x = x * spatial_att

        return x
###############################################


#####################MobileNetV3 + Attention#####################
class MobileNetV3(nn.Module):
    def __init__(self):
        super(MobileNetV3, self).__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT
        base_model = mobilenet_v3_small(weights=weights)

        self.features = base_model.features  # [B, 576, H, W]

     
        self.cbam = CBAM(channels=576)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        return x

# class MobileNetV3Lite(nn.Module):
#     def __init__(self):
#         super(MobileNetV3Lite, self).__init__()
#
#         weights = MobileNet_V3_Small_Weights.DEFAULT
#         base_model = mobilenet_v3_small(weights=weights)
#
#   
#         self.features = nn.Sequential(*list(base_model.features.children())[:10])  # 输出通道为96
#         print(self.features(torch.randn(1, 3, 224, 224)).shape)
#
#
#         #self.cbam = CBAM(channels=96)
#
#     def forward(self, x):
#         x = self.features(x)
#         #x = self.cbam(x)
#         return x
########################AlexNetv2-mixattentiob-depthwise############################
# 混合注意力（MixAttention)
class MixAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(MixAttention, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid_channel(avg_out + max_out)
        x = x * ca

        # 空间注意力
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        sa = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg, max_], dim=1)))
        x = x * sa
        return x



class AlexNetV2_MixAttn_DW(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2_MixAttn_DW, self).__init__()
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        

    
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, stride=1, groups=96),
            nn.Conv2d(96, 256, kernel_size=1, stride=1),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.attn2 = MixAttention(256)

     
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.attn3 = MixAttention(384)

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.attn4 = MixAttention(384)

        
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, kernel_size=1, stride=1))
        

# class AlexNetV2_MixAttn_DW(_AlexNet):
#     output_stride = 4
#
#     def __init__(self):
#         super(AlexNetV2_MixAttn_DW, self).__init__()
#         # Conv1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 96, kernel_size=11, stride=2),
#             _BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2))
#         self.attn1 = MixAttention(96)
#
#         # Conv2 - Depthwise + Pointwise
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(96, 96, kernel_size=5, stride=1, groups=96),  # Depthwise
#             nn.Conv2d(96, 256, kernel_size=1, stride=1),             # Pointwise
#             _BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=1))
#         self.attn2 = MixAttention(256)
#
#         # Conv3
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(256, 384, kernel_size=3, stride=1),
#             _BatchNorm2d(384),
#             nn.ReLU(inplace=True))
#         self.attn3 = MixAttention(384)
#
#         # Conv4 - Depthwise + Pointwise
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=384),
#             nn.Conv2d(384, 384, kernel_size=1, stride=1),
#             _BatchNorm2d(384),
#             nn.ReLU(inplace=True))
#         self.attn4 = MixAttention(384)
#
#         # Conv5 - Depthwise + Pointwise
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=384),
#             nn.Conv2d(384, 32, kernel_size=1, stride=1))
#         self.attn5 = MixAttention(32)






