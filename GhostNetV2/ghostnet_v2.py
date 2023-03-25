import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp(0., 6.).div_(6.)
    else:
        return F.relu(x + 3.) / 6.
    
# SE Layer
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25, reduce_base_channels=None, act_layer=nn.ReLU, 
                 gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_channels = _make_divisible((reduce_base_channels or in_channels) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)

        return x
    
# ConvBnAct
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)

        return x

# Ghost Module
class GhostModuleV2(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=1, ratio=2, 
                 dw_size=3, stride=1, relu=True, mode=None, args=None):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
            self.out_channels = out_channels
            init_channels = math.ceil(out_channels / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(input_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
        elif self.mode in ['attn']:
            self.out_channels = out_channels
            init_channels = math.ceil(out_channels / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(input_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 

            self.short_conv = nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)

            return out[:, :self.out_channels, :, :]

        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)

            return out[:, :self.out_channels, :, :] * \
                F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-2], out.shape[-1]), mode='nearest')