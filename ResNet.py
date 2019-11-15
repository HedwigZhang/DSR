###################
# The implemation of ResNeXt
###################

import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow.keras import layers
#Input, Conv2d, AveragePooling2D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

def bn_relu_conv(input_batch, ch, k_size, st, name):
    '''
    realize conv + bn + relu operation
    :param input_batch: 4D tensor
    :param ch: 1D int, the channles
    :param k_size: 1D int, the size of convolution kernel
    '''
    x = layers.BatchNormalization()(input_batch)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(ch, k_size, strides=(st, st), padding = "same", kernel_regularizer = l2(0.002), use_bias = False, name = name)(x) # , activity_regularizer=l1(0.0002) , kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002)
    return x

def bn_relu_conv_valid(input_batch, ch, k_size, st, name):
    '''
    realize conv + bn + relu operation
    :param input_batch: 4D tensor
    :param ch: 1D int, the channles
    :param k_size: 1D int, the size of convolution kernel
    '''
    x = layers.BatchNormalization()(input_batch)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(ch, k_size, strides=(st, st), padding = "valid", kernel_regularizer = l2(0.002), use_bias = False, name = name)(x) # , activity_regularizer=l1(0.0002)  kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002),
    return x

def bn_dw_conv_valid(inputs, k_size, strs, name):
    '''
    the depthwise convolution, which also contains two operation: BN and dwconv.
    :param inputs: 3D tensor, input data
    :param k_size: 1D integer, the size of kernel
    :param strs: 1D integer, the stride for convolution
    :return: 3D tensor.
    '''
    x = layers.BatchNormalization(axis=-1, name='{}_bn'.format(name))(inputs)
    x = layers.DepthwiseConv2D(kernel_size = k_size, strides=(strs, strs), padding='valid', depthwise_regularizer = l2(0.002), use_bias = False, name='{}_dwconv'.format(name))(x) #  depthwise_regularizer = l2(0.002), use_bias = False,
    return x
# def squeeze_excitation_res_layer(inputs, out_dim, name):
#     '''
#     the squeeze_excitation
#     '''
#     squeeze = layers.GlobalAveragePooling2D(name = '{}_gap'.format(name))(inputs)
#     #excitation = layers.Activation('sigmoid')(squeeze)
#     # excitation = layers.Dense(units = out_dim, activation="sigmoid", kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002), name = '{}_ex'.format(name))(squeeze)
#     excitation = layers.Dense(units = out_dim // 16, activation="relu", kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002), name = '{}_sq'.format(name))(squeeze)
#     excitation = layers.Dense(units = out_dim, activation="sigmoid", kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002), name = '{}_ex'.format(name))(excitation)
#     excitation = layers.Reshape((1,1,out_dim))(excitation)

#     scale = layers.Multiply(name = '{}_scale'.format(name))([inputs,excitation])

#     scale = layers.Add(name = '{}_res'.format(name))([inputs, scale])
#     return scale

def DS_block(inputs, name):
    '''
    The DSE block. For different size, the ksize is different.
    '''
    size = inputs.get_shape().as_list()[1]
    if size == 32:
        k1 = 7
        st = 4
        k2 = 7
        squeeze = bn_dw_conv_valid(inputs, k1, st, name = '{}_dw1'.format(name))
        squeeze = bn_dw_conv_valid(squeeze, k2, k2, name = '{}_dw2'.format(name))
    if size == 16:
        k1 = 5
        st = 2
        k2 = 6
        squeeze = bn_dw_conv_valid(inputs, k1, st, name = '{}_dw1'.format(name))
        squeeze = bn_dw_conv_valid(squeeze, k2, k2, name = '{}_dw2'.format(name))
    else:
        k = size
        squeeze = bn_dw_conv_valid(inputs, k, k, name = '{}_dw'.format(name))
    
    scale = layers.Activation('sigmoid')(squeeze)
    scale = layers.Multiply(name = '{}_scale'.format(name))([inputs,scale])
    scale = layers.Add(name = '{}_res'.format(name))([inputs, scale])
    return scale
    

def bottleneck_block_stage1(inputs, out_ch, name):
    '''
    This bottleneck of the ResNet
    :param inputs: 3D tesnor, the input data
    :param out_ch: 1D integer, the output channels
    return 3D tensor
    '''
    ############### here the in_ch = 64, out_ch = 256. Hence the input is 32*32*64
    sub_ch = out_ch // 4
    ############ ResNet
    res = bn_relu_conv(inputs, out_ch, 1, 1, name = '{}_res'.format(name)) #### 32*32*256
    ############## conv
    br = bn_relu_conv(inputs, sub_ch, 1, 1, name = '{}_c1'.format(name))  #### 32*32*64
    br = bn_relu_conv(br, sub_ch, 3, 1, name = '{}_c2'.format(name))      #### 32*32*64
    br = bn_relu_conv(br, out_ch, 1, 1, name = '{}_c3'.format(name))      #### 32*32*256
    ############## The DS block
    br = DS_block(br, name = '{}_ds'.format(name))
    ############## The resdiaul
    out = layers.Add(name='{}_add'.format(name))([br, res])  #### 32*32*256
    return out

def bottleneck_block_stage_2_3(inputs, out_ch, name):
    '''
    This block realize ResNeXt in (b)
    :param inputs: 3D tesnor, the input data
    :param out_ch: 1D integer, the output channels
    return 3D tensor
    '''
    ############### here the in_ch = 64, out_ch = 256. Hence the input is 32*32*64
    sub_ch = out_ch // 4
    ############ res 
    # res = bn_relu_conv(inputs, out_ch, 1, 1, name = '{}_res'.format(name))
    res = layers.AveragePooling2D(2)(inputs)
    res = layers.Lambda(data_pad)(res)
    ############## conv
    br = bn_relu_conv(inputs, sub_ch, 1, 1, name = '{}_c1'.format(name))  #### 14*14*128, 7*7*256
    br = bn_relu_conv(br, sub_ch, 3, 2, name = '{}_c2'.format(name))      #### 14*14*128, 7*7*256
    br = bn_relu_conv(br, out_ch, 1, 1, name = '{}_c3'.format(name))      #### 14*14*512, 7*7*1024
    ############## The DS block
    br = DS_block(br, name = '{}_ds'.format(name))
    ############## The resdiaul
    out = layers.Add(name='{}_add'.format(name))([br, res])
    return out


def bottleneck_block(inputs, out_ch, name):
    '''
    This block realize ResNeXt in (b)
    :param inputs: 3D tesnor, the input data
    :param out_ch: 1D integer, the output channels
    return 3D tensor
    '''
    sub_ch = out_ch // 4
    ### the first and second layer is convolution layer
    br = bn_relu_conv(inputs, sub_ch, 1, 1, name = '{}_c1'.format(name))  #### h*w*sub_ch
    br = bn_relu_conv(br, sub_ch, 3, 1, name = '{}_c2'.format(name))      #### h*w*sub_ch
    br = bn_relu_conv(br, out_ch, 1, 1, name = '{}_c3'.format(name))      #### h*w*out_ch
    ############## The DS block
    br = DS_block(br, name = '{}_ds'.format(name))
    ############## The resdiaul
    out = layers.Add(name='{}_add'.format(name))([br, inputs])
    return out

def data_pad(x):
    ch = x.get_shape().as_list()[-1]
    sub_ch = ch // 2
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [sub_ch, sub_ch]])
    return x

def resnet50():
    '''
    realize the Residual inception Net
    '''
    # Input
    input = layers.Input((32,32,3))   # 32 * 32 * 3
    ## first conv layer
    x = bn_relu_conv(input, 64, 3, 1, name = 'C0')  # 32 * 32 * 64
    ############# stage1
    x = bottleneck_block_stage1(x, 256, name = 'stage1') # 32 * 32 * 256
    for j in range(2):
        x = bottleneck_block(x, 256, name = 'resnext1_'+ str(j))

    ############# stage2
    x = bottleneck_block_stage_2_3(x, 512, name = 'stage2') # 16 * 16 * 512
    for j in range(3):
        x = bottleneck_block(x, 512, name = 'resnext2_'+ str(j))

    ############# stage3
    x = bottleneck_block_stage_2_3(x, 1024, name = 'stage3') # 8 * 8 * 1024
    for j in range(5):
        x = bottleneck_block(x, 1024, name = 'resnext3_'+ str(j))

    ############# stage4
    x = bottleneck_block_stage_2_3(x, 2048, name = 'stage4') # 4 * 4 * 2048
    for j in range(2):
        x = bottleneck_block(x, 2048, name = 'resnext4_'+ str(j))

    x = layers.GlobalAveragePooling2D()(x)
    # 1 * 1 * 128
    x = layers.Flatten()(x)
    x = layers.Dense(1000, kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002), activation="softmax")(x) #, activity_regularizer=l1(0.0002) kernel_regularizer = l2(0.002), bias_regularizer = l2(0.002)
    model = Model(input, x)
    return model