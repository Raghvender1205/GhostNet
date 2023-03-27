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
        
# Bottlneck V2
class GhostNetBottleneckV2(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dw_kernel_size=3, 
                 stride=1, se_ratio=0., layer_id = 0, args=None):
        super(GhostNetBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Pointwise Expansion
        if layer_id <= 1: # lOriginal  [layer_id = None]
            self.ghost1 = GhostModuleV2(in_channels, mid_channels, relu=True, mode='attn', args=args)
        else:
            self.ghost1 = GhostModuleV2(in_channels, mid_channels, relu=True, mode='attn', args=args)
        
        # Depthwise Convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size-1)//2, groups=mid_channels, bias=False)
            self.bn_dw = nn.BatchNorm1d(mid_channels)

        # Squeeze-and-Excitation
        if has_se:
            self.se = SqueezeExcite(mid_channels, se_ratio=se_ratio)
        else:
            self.se = None
        self.ghost2 = GhostModuleV2(mid_channels, out_channels, relu=False, mode='original', args=args)

        # Shortcut
        if (in_channels == out_channels and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride=stride, 
                          padding=(dw_kernel_size-1)//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)

        return x
    

# Model
class GhostNetV2(nn.Module):
    def __init__(self, cfgs, n_classes=1000, width_mutliplier=1.0, dropout=0.2, 
                 block=GhostNetBottleneckV2, args=None):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # Build First layer
        output_channel = _make_divisible(16 * width_mutliplier, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # Build Inverted Residual Blocks
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width_mutliplier, 4)
                hidden_channel = _make_divisible(exp_size * width_mutliplier, 4)
                if block == GhostNetBottleneckV2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio, layer_id=layer_id, args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))
        output_channel = _make_divisible(exp_size * width_mutliplier, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel 
        self.blocks = nn.Sequential(*stages) 

        # Building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, n_classes)
    
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x
    
def ghostnetv2(**kwargs):
    cfgs = [   
        # k, t, c, SE, s 
        [[3,  16,  16, 0, 1]],
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]

    """
    return GhostNetV2(cfgs, n_classes=kwargs['n_classes'], 
                    width_multiplier=kwargs['width'], 
                    dropout=kwargs['dropout'],
                    args=kwargs['args'])
    """
    return GhostNetV2(cfgs, **kwargs)

if __name__ == '__main__':
    model = ghostnetv2()
    print(model.parameters)