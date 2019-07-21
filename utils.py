# imort modules
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages



# functions


def get_mask_wh(var, mask_shape):
    """
    Get conv kernel width height mask
    var: the variable to be masked
    mask_shape: int or tuple, represents the center (w,h) kernel to extract
    Return the widht height mask
    """
    # TODO: channel first or channel last
    shape = var.shape
    assert len(shape) == 4

    # initial mask
    mask = np.zeros(shape)

    # mask_shape (width,height)
    if isinstance(mask_shape,int):
        mask_shape = (mask_shape,mask_shape)

    # calculate mask position
    start_w = (shape[0].value - mask_shape[0])//2
    start_h = (shape[1].value - mask_shape[1])//2
    end_w = start_w + mask_shape[0]
    end_h = start_h + mask_shape[1]
    mask[start_w:end_w,start_h:end_h,:,:] = 1.0

    # convert to tensor
    mask = tf.convert_to_tensor(mask,tf.float32)
    return mask


def get_mask_c(var, mask_shape):
    """
    Get conv kernel channel mask
    var: the variable to be masked
    mask_shape: int represent the first N channels to extract
    Return the channel mask
    """
    # TODO: channel first or channel last
    shape = var.shape
    assert len(shape) == 4
    assert mask_shape <= var.shape[2]

    # initial mask
    mask = np.zeros(shape)

    mask[:,:,:mask_shape,:] = 1.0

    # convert to tensor
    mask = tf.convert_to_tensor(mask,tf.float32)
    return mask


def masked_kernel(name,kernel_shape, wh_mask_shape, c_mask_shape,dtype = tf.float32):
    """
    Get the masked convolution
    name: the variable name
    kernel_shape: the max kernel shape
    wh_mask_shape: the width height mask shape
    c_mask_shape: the channel mask shape
    dtype: the kernel data type
    return the masked convolution kernel and the corresponding select placeholder
    """
    # width height select & channel select
    wh_select_ph = tf.placeholder(tf.int32)
    c_select_ph = tf.placeholder(tf.int32)

    # super kernel
    full_conv_var = tf.get_variable(name = name,shape=kernel_shape,dtype=dtype)

    # get width height mask
    wh_mask = tf.case([(tf.equal(wh_select_ph,i), lambda shape=shape :get_mask_wh(full_conv_var,shape)) for i,shape in enumerate(wh_mask_shape)])
    # get channel mask
    c_mask = tf.case([(tf.equal(c_select_ph,i), lambda shape=shape:get_mask_c(full_conv_var,shape)) for i,shape in enumerate(c_mask_shape)])

    # masked convolution
    masked_kernel = full_conv_var * wh_mask * c_mask

    return masked_kernel, wh_select_ph, c_select_ph


def batch_norm(x, is_training, name="bn", decay=0.9, epsilon=1e-5,
               data_format="NHWC"):
  """
  reference from https://github.com/melodyguan/enas/blob/master/src/cifar10/image_ops.py
  """
  if data_format == "NHWC":
    shape = [x.get_shape()[3]]
  elif data_format == "NCHW":
    shape = [x.get_shape()[1]]
  else:
    raise ValueError("Unknown data_format {}".format(data_format))

  with tf.variable_scope(name, reuse=None if is_training else True):
    offset = tf.get_variable(
      "offset", shape,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable(
      "scale", shape,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    moving_mean = tf.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    moving_variance = tf.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))

    if is_training:
      x, mean, variance = tf.nn.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      update_mean = moving_averages.assign_moving_average(
        moving_mean, mean, decay)
      update_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
      with tf.control_dependencies([update_mean, update_variance]):
        x = tf.identity(x)
    else:
      x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                       variance=moving_variance,
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
  return x

def masked_conv(feature,name,kernel_shape,wh_mask_shape,c_mask_shape,strides,padding='SAME',
                bn = True,is_training=True,decay=0.9,epsilon=1e-5,
                activation = True,dtype = tf.float32,data_format="NHWC"):
    """
    Convolution using masked kernel
    name: the variable name
    kernel_shape: the max kernel shape
    wh_mask_shape: the width height mask shape
    c_mask_shape: the channel mask shape
    strides: the convolution stride
    padding: the convolution padding
    data_format: NCHW or NHWC
    dtype: the kernel data type
    return the masked convolution result and the corresponding select placeholder
    """
    kernel, wh_select_ph, c_select_ph = masked_kernel(name,kernel_shape,wh_mask_shape,c_mask_shape,dtype)
    if isinstance(strides,int):
        strides = [strides,strides]
    if data_format == "NHWC":
        strides = [1,strides[0],strides[1],1]
    elif data_format == "NCHW":
        strides = [1,1,strides[0],strides[1]]
        pass
    else:
        raise ValueError("Unknown data_format {}".format(data_format))
    with tf.variable_scope(name):
        out = tf.nn.conv2d(feature,filter=kernel,strides=strides,padding=padding,data_format = data_format,name = name)
        if bn:
            out = batch_norm(out,is_training,decay = decay,epsilon=epsilon,data_format=data_format)
        if activation:
            out = tf.nn.leaky_relu(out, alpha=0.1)
        return out,wh_select_ph,c_select_ph


def dynamic_depth_residual_block(input_data,name,kernel_shape,wh_mask_shape,c_mask_shape,
                   bn = True, is_training = True,decay=0.9,epsilon=1e-5,
                   activation = True,dtype = tf.float32,data_format="NHWC",
                   max_num_layer=2):

    short_cut = input_data
    depth_select_ph = tf.placeholder(tf.int32)
    candidate_layers = []
    select_phs = []
    with tf.variable_scope(name):
        for i in range(max_num_layer):
            input_data,wh_select_ph,c_select_ph = masked_conv(input_data,"conv"+str(i),kernel_shape,wh_mask_shape,c_mask_shape,1,'SAME',
                                     bn = bn,is_training=is_training,decay=decay,epsilon =epsilon,
                                     activation = activation, dtype = dtype, data_format = data_format)
            candidate_layers.append((tf.equal(depth_select_ph,i),lambda:input_data))
            select_phs.append(wh_select_ph)
            select_phs.append(c_select_ph)
        input_data = tf.case(candidate_layers)
        residual_output = input_data + short_cut
        select_phs.append(depth_select_ph)

    return residual_output,select_phs



def depthwise_conv():
    pass

def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    conv1,wh_select_ph,c_select_ph = masked_kernel("test",[7,7,128,256],wh_mask_shape = [1,3,5,7], c_mask_shape = [16,32,48,64,80,96,112,128])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print (sess.run(conv1,{wh_select_ph:2,c_select_ph:7}))
