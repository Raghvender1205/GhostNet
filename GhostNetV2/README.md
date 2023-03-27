# GhostNetV2: Enhance Cheap Operation with Long Range Attention
In this, a new Attention Mechanism (`dubbed DFC attention`) is proposed to capture `long-range` spatial information while keeping the implementation efficiency.
1. Only `FC` layers are involved in generating the `attention` maps.
2. An `FC` layer is decomposed into horizontal and vertical `FC` to aggregate pixels in `2D` feature map.
3. These 2 FC layers involve pixels in a long range along their respective directions.
4. Then, stack them which would produce a global `receptive` field.
5. Also, we revisit the `bottleneck` of GhostNet and enhance the intermediate features with `DFC` attention.
## Approach
### DFC Attention
A desired attention is expected to have the following properties
1. `Long Range`:- It is important to capture spatial information to enhance the representation ability. 
2. `Deployement-Efficient`:- The attention module should be efficient to avoid slow inference.
3. `Concept-Simple`: The attention should be conceptually simple to keep the model's generalization

`Self-Attention` operations can model the `long-range` dependence but they are not `deployement-efficient`. Compared to them `Fully-Connected (FC)` layers with fixed weights are simple to implement.


![Bottleneck](https://github.com/Raghvender1205/GhostNet/blob/master/GhostNetV2/docs/bottleneckV1vsV2.png?raw=true)

The figure above is an illustration of the `GhostNet` and `GhostNetV2` block. `GhostBlock` is an inverted residual bottleneck containing two Ghost modules, where `DFC` Attention enhances the expanded features to improve expressive ability.

### Enhancing Ghost Module
In `GhostNet`, only half of the features in the Ghost Module interact with other pixels, which damages its abilty to capture `spatial` information. Hence, we use `DFC` attention to enhance the Ghost Module's output feature $Y$ for capturing `long-range` dependence among other different spatial pixels.

In this, the `input` feature is sent to two branches.
1. Ghost Module to produce output feature $Y$
2. `DFC` module to generate attention map

As in `SelfAttention`, linear tranformation layers are used to transform an input feature into `query` and `key` to calculate attention map. Similarly, a `1 x 1` Conv is implemented to convert module's input $X$ into `DFC's` input $Z$. The final output would then be the product of two branch's `output`.

`Ghost Module` and `DFC Attention` are two parallel branches which extract information from different perspective of same input. The `output` is an `element-wise product`, which would contain information from both `features`. The calculation of each `attention` value involves patches of large range so that the output feature can contain information from these patches.

### Feature Downsampling
As `GhostModule` is an efficient operation, directly paralleling the `DFC Attention` will result in more computation cost.

So, we reduce the feature's size by `down-sampling` it both horizontally and vertically so that all `DFC` attention can be conducted on smaller features. We use the following for `down-sampling` and `up-sampling`.
1. Average Pooling
2. Bilinear Interpolation

We also deploy `sigmoid` on the downsampled features to accelerate practical inference.

### GhostNetV2 Bottleneck
In this, a `DFC` attention branch in parallel with `Ghost Module` is used to enhance the expanded features. Then, these enhanced features are sent to the second Ghost Module for producing output features.

## Links
1. https://arxiv.org/pdf/2211.12905.pdf [Original Paper]
2. https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnetv2_pytorch/model/ghostnetv2_torch.py [Implementation]

## Model Architecture

```
<bound method Module.parameters of GhostNetV2(
  (conv_stem): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU(inplace=True)
  (blocks): Sequential(
    (0): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
            (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(16, 16, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=16, bias=False)
            (3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=16, bias=False)
            (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
            (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
    )
    (1): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(16, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(16, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(48, 48, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=48, bias=False)
            (3): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(48, 48, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=48, bias=False)
            (5): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (conv_dw): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
        (bn_dw): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=12, bias=False)
            (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential(
          (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Conv2d(16, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (2): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(24, 36, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=36, bias=False)
            (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(72, 72, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=72, bias=False)
            (3): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(72, 72, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=72, bias=False)
            (5): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(72, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=12, bias=False)
            (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
    )
    (3): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(24, 36, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=36, bias=False)
            (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(72, 72, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=72, bias=False)
            (3): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(72, 72, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=72, bias=False)
            (5): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (conv_dw): Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
        (bn_dw): BatchNorm1d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SqueezeExcite(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (conv_reduce): Conv2d(72, 20, kernel_size=(1, 1), stride=(1, 1))
          (act1): ReLU(inplace=True)
          (conv_expand): Conv2d(20, 72, kernel_size=(1, 1), stride=(1, 1))
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(72, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
            (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential(
          (0): Conv2d(24, 24, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=24, bias=False)
          (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Conv2d(24, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (4): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(40, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(60, 60, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=60, bias=False)
            (1): BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(120, 120, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=120, bias=False)
            (3): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(120, 120, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=120, bias=False)
            (5): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (se): SqueezeExcite(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (conv_reduce): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
          (act1): ReLU(inplace=True)
          (conv_expand): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
            (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
    )
    (5): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)
            (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(240, 240, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=240, bias=False)
            (3): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(240, 240, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=240, bias=False)
            (5): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        (bn_dw): BatchNorm1d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential(
          (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=40, bias=False)
          (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Conv2d(40, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (6): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(80, 100, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(100, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=100, bias=False)
            (1): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(80, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(200, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=200, bias=False)
            (3): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(200, 200, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=200, bias=False)
            (5): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(200, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
      (1): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(80, 92, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(92, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=92, bias=False)
            (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(184, 184, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=184, bias=False)
            (3): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(184, 184, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=184, bias=False)
            (5): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(184, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
      (2): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(80, 92, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(92, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=92, bias=False)
            (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(184, 184, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=184, bias=False)
            (3): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(184, 184, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=184, bias=False)
            (5): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(184, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
            (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
      (3): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(80, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
            (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(480, 480, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=480, bias=False)
            (3): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(480, 480, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=480, bias=False)
            (5): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (se): SqueezeExcite(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (conv_reduce): Conv2d(480, 120, kernel_size=(1, 1), stride=(1, 1))
          (act1): ReLU(inplace=True)
          (conv_expand): Conv2d(120, 480, kernel_size=(1, 1), stride=(1, 1))
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(480, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=56, bias=False)
            (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential(
          (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
          (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Conv2d(80, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (4): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=336, bias=False)
            (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(672, 672, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=672, bias=False)
            (3): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(672, 672, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=672, bias=False)
            (5): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (se): SqueezeExcite(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (conv_reduce): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
          (act1): ReLU(inplace=True)
          (conv_expand): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(672, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=56, bias=False)
            (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
    )
    (7): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=336, bias=False)
            (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(672, 672, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=672, bias=False)
            (3): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(672, 672, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=672, bias=False)
            (5): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
        (bn_dw): BatchNorm1d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SqueezeExcite(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (conv_reduce): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
          (act1): ReLU(inplace=True)
          (conv_expand): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(672, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential(
          (0): Conv2d(112, 112, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=112, bias=False)
          (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Conv2d(112, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (8): Sequential(
      (0): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(960, 960, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=960, bias=False)
            (3): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(960, 960, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=960, bias=False)
            (5): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
      (1): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(960, 960, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=960, bias=False)
            (3): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(960, 960, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=960, bias=False)
            (5): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (se): SqueezeExcite(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (conv_reduce): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
          (act1): ReLU(inplace=True)
          (conv_expand): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
      (2): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(960, 960, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=960, bias=False)
            (3): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(960, 960, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=960, bias=False)
            (5): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
      (3): GhostNetBottleneckV2(
        (ghost1): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (cheap_operation): Sequential(
            (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (short_conv): Sequential(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(960, 960, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=960, bias=False)
            (3): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): Conv2d(960, 960, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=960, bias=False)
            (5): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (se): SqueezeExcite(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (conv_reduce): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
          (act1): ReLU(inplace=True)
          (conv_expand): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
        )
        (ghost2): GhostModuleV2(
          (gate_fn): Sigmoid()
          (primary_conv): Sequential(
            (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
          (cheap_operation): Sequential(
            (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
            (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Sequential()
          )
        )
        (shortcut): Sequential()
      )
    )
    (9): Sequential(
      (0): ConvBnAct(
        (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): ReLU(inplace=True)
      )
    )
  )
  (global_pool): AdaptiveAvgPool2d(output_size=(1, 1))
  (conv_head): Conv2d(960, 1280, kernel_size=(1, 1), stride=(1, 1))
  (act2): ReLU(inplace=True)
  (classifier): Linear(in_features=1280, out_features=1000, bias=True)
)>
```