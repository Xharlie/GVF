import tensorflow as tf
import tf_util
import numpy as np
# from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
# K.set_image_data_format("channels_first")
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(inputs, channel_size=(512, 256, 128, 64, 64), pool_size=(2, 2, 2), deconvolution=False, depth=4, batch_normalization=False, instance_normalization=False, activation_lst=None, res=True, training=True):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).

    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    current_layer = inputs

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth):
        if res:
            if current_layer.get_shape()[-1].value != channel_size[layer_depth]:
                previous_layer = create_convolution_block(n_filters=channel_size[layer_depth], input_layer=current_layer, batch_normalization=batch_normalization, instance_normalization=instance_normalization, kernel=(1,1,1), activation=None, training=training)
            else:
                previous_layer = current_layer
            up_pre = get_up_convolution(previous_layer,pool_size=pool_size, deconvolution=False, n_filters=channel_size[layer_depth])
        else:
            up_pre = 0
        current_layer = get_up_convolution(current_layer,pool_size=pool_size, deconvolution=deconvolution, n_filters=channel_size[layer_depth], batch_normalization=batch_normalization,activation=activation_lst[layer_depth], training=training)
        print("current_layer",layer_depth, channel_size[layer_depth], current_layer.get_shape().as_list())
        if not deconvolution:
            current_layer = create_convolution_block(n_filters=channel_size[layer_depth], input_layer=current_layer, batch_normalization=batch_normalization, instance_normalization=instance_normalization, activation=activation_lst[layer_depth],training=training)
        current_layer = current_layer + up_pre

    final_convolution = Conv3D(channel_size[depth], (1, 1, 1))(current_layer)
    act = get_activation(activation_lst[depth])(final_convolution)
    return act

def get_activation(act):
    if isinstance(act, str):
        return Activation(act)
    else:
        return act()


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None, padding='same', strides=(1, 1, 1), instance_normalization=False, training=True):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        print("pre bn layer.shape",layer.get_shape().as_list())
        layer = BatchNormalization(axis=-1)(layer, training = training)
    elif instance_normalization:
        # try:
        # except ImportError:
        #     raise ImportError("Install keras_contrib in order to use instance normalization."
        #                       "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=-1)(layer, training = training)
    if activation is None:
        return layer
    else:
        return get_activation(activation)(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(inputs, n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), deconvolution=False,activation=None,batch_normalization=False,instance_normalization=False,training=False):
    if deconvolution:
        layer = Deconvolution3D(filters=n_filters, kernel_size=kernel_size, strides=strides)(inputs)
        if batch_normalization:
            layer = BatchNormalization(axis=-1)(layer, training=training)
        elif instance_normalization:
            layer = InstanceNormalization(axis=-1)(layer, training=training)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return get_activation(activation)(layer)
    else:
        return UpSampling3D(size=pool_size)(inputs)

