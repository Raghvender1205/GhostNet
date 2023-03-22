import torch
import torch.nn as nn
import math

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, expansion_ratio=2, 
                 dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        # Compute channels for both primary and secondary Conv
        init_channels = math.ceil(out_channels / expansion_ratio)
        new_channels = init_channels * (expansion_ratio - 1)

        # Primary Conv + BatchNorm, ReLU
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # Secondary Conv + BatchNorm + ReLU
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # Stack these two
        out = torch.cat([x1, x2], dim=1)

        return out[:, :self.out_channels, :, :]
    
# Squeeze Excitation Layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)

        return x * y
    
# Depthwise Conv (DWConv)
def dw_conv(input_channels, out_channels, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(input_channels, out_channels, kernel_size, stride, kernel_size//2, groups=input_channels, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


# Ghost Bottlneck
class GhostBottleneck(nn.Module):
    def __init__(self, input_channels, hidden_dim, out_channels, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # Pointwise Conv
            GhostModule(input_channels, hidden_dim, kernel_size=1, relu=True),
            # depthwise Conv
            dw_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride == 2 else nn.Sequential(),
            # Squeeze Excitation
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, out_channels, kernel_size=1, relu=False),
        )

        if stride == 1 and input_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                dw_conv(input_channels, input_channels, kernel_size, stride, relu=False),
                nn.Conv2d(input_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

def _make_divisible(v, divisor, min_value=None):
    """
    Ensures all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Round down should not go more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    
    return new_v

