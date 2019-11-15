# DSR
Depthwise squeeze and refinement

The Squeeze operation applies two depthwise convolution to capture the global information of each feature map, which effectively avoids the information loss caused by global pooling.

The Refinement applies the Sigmoid function to evaluate each feature map via the obtained global information.
