# GhostNet: More Features From Cheap Operations
In this, we use different resources and their variants of implementations of `GhostNet` to understand the architecture and performance of `GhostNet`.

## Approach

<img src="https://blog.paperspace.com/content/images/2020/06/Capture.PNG"/>

In `GhostNet`, we introduce a module which utilizes a few `small filters` to generate `more` feature maps from the original `Conv` layer and then develop `GhostNet` with an extremely efficient architecture and performance.

A Conv Layer is defined by number of input channels $C$, which outputs a `tensor` of $C'$. These are the `feature maps` and many redundant copies of it remain. So, GhostNet generate `x%` of the total `output feature maps`, while the remaining are created by a `cheap linear` operation.

This `cheap linear` operation results in massive reduction in model parameters and `FLOPs` while retaining the same performance as of the original baseline model. 

`GhostNet` uses `Depthwise Convolution (DWConv)` as its cheap linear transformation

### Build Efficient CNNs
Using the GhostModules, Ghost Bottleneck `G-bneck` is introduced which has two stacked Ghost modules.
1. First Ghost module acts as an `expansion` layer increasing the number of channels known as `expansion ratio`.
2. Second Ghost module reduces the number of channels to match the `shortcut path`.

Now, the `shortcut` is connected between the `inputs` and `outputs` of these two Ghost Modules. `Shortcut` path is implemented by 
1. DownSampling layer
2. Depthwise Convolution.
3. In primary convolution, Ghost Module is a `Pointwise Convolution`.

----------------
Following on the `G-Bottleneck`, a GhostNet is presented in which `MobileNetV3` architecture is followed. 

We replace the `Bottleneck` part of the `MobileNetV3` and replace it with our `GhostBottleneck`

### Model Architecture
- First layer is a `Conv` layer with 16 filters
- Followed by a series of `Ghost Bottlenecks` with gradually increased channels.
- These `G-Bnecks` are grouped into different stages according to the size of their `input feature map`.
- All `G-Bnecks` are applied with `stride=1` except the last one in each stage is with `stride=2`.
- At last a `GlobalAveragePool2d` and a `Conv` are used to transform the feature maps into a feature vector.

In contrast to `MobileNetV3`, $hard-swish$ nonlinearity is not used due to its `large latency`.

### Width Multiplier
To customize the Model for some scenarios, we multiply a factor $\alpha$ on the number of channels uniformly at each layer. 

Factor $\alpha$ is called a `Width Multiplier` as it can change the width of the entire network. It can control
1. Model size
2. Computational cost 

Usually $\alpha$ leads to lower latency and lower performance.

For more information refer to these links 
1. Official Implementation:- https://github.com/huawei-noah/Efficient-AI-Backbones
2. Original Arxiv Paper: https://arxiv.org/pdf/1911.11907.pdf
3. Paperspace Blog:- https://blog.paperspace.com/ghostnet-cvpr-2020/
4. Review of GhostNet:- https://sh-tsang.medium.com/review-ghostnet-more-features-from-cheap-operations-1784f3bbc2b

## TODO
1. Review and understand `GhostNetV2`.

`GhostNetV2` was also proposed in <b>CVPR 2022</b>
- https://arxiv.org/abs/2211.12905 [GhostNetV2 Paper]
- https://github.com/likyoo/GhostNetV2-PyTorch [Implementation]