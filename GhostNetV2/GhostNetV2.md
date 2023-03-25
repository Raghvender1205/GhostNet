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