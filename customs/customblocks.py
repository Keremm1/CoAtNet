import tensorflow as tf
from keras import layers
import tensorflow_addons as tfa
from .customlayers import Conv2DNoBias, ConvBN2D, MHSAWithMultiHeadRelativePositionEmbedding, DepthwiseConv2DNoBias

class SEBlock(layers.Layer):
    def __init__(self, se_ratio=0.25, divisor=8, limit_round_down=0.9, activation="relu", use_bias=True, use_conv=True, **kwargs):
        super().__init__()
        self.se_ratio = se_ratio
        self.divisor = divisor
        self.limit_round_down = limit_round_down
        self.activation = activation
        self.use_bias = use_bias
        self.use_conv = use_conv

    def build(self, input_shape):
        channel_axis = -1
        h_axis, w_axis = [1, 2]

        hidden_activation, output_activation = self.activation if isinstance(self.activation, (list, tuple)) else (self.activation, "sigmoid")
        
        self.filters = filters = input_shape[channel_axis]
        reduction = self._make_divisible(
            filters * self.se_ratio, self.divisor, limit_round_down=self.limit_round_down
        )

        self.reduce_mean_layer = layers.Lambda(
            lambda x: tf.math.reduce_mean(x, axis=[h_axis, w_axis], keepdims=True if self.use_conv else False)
        )
        if self.use_conv:
            self.se_conv1 = layers.Conv2D(
                reduction, kernel_size=1, use_bias=self.use_bias
            )
        else:
            self.se_dense1 = layers.Dense(
                reduction, use_bias=self.use_bias
            )

        self.hidden_activation_layer = layers.Activation(hidden_activation)

        if self.use_conv:
            self.se_conv2 = layers.Conv2D(
                filters, kernel_size=1, use_bias=self.use_bias
            )
        else:
            self.se_dense2 = layers.Dense(
                filters, use_bias=self.use_bias
            )

        self.output_activation_layer = layers.Activation(output_activation)

    def call(self, inputs):
        x = self.reduce_mean_layer(inputs)
        if self.use_conv:
            x = self.se_conv1(x)
        else:
            x = self.se_dense1(x)
        x = self.hidden_activation_layer(x)
        if self.use_conv:
            x = self.se_conv2(x)
        else:
            x = self.se_dense2(x)
        x = self.output_activation_layer(x)
        x = x if self.use_conv else tf.reshape(
            x,
            [-1, 1, 1, self.filters],
        )
        return layers.Multiply()([inputs, x])

    @staticmethod
    def _make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)

        if new_v < limit_round_down * vv:
            new_v += divisor
        return new_v


class MSHABlock(layers.Layer):
    def __init__(self, output_channel, stride=1, head_dimension=32):
        super().__init__()
        stride = stride[0] if isinstance(stride, tuple) else stride
        
        self.output_channel = output_channel
        self.stride = stride
        self.head_dimension = head_dimension
        

    def build(self, input_shape):
        self.preact_layer_norm = layers.LayerNormalization(epsilon=1e-5)
        if self.stride != 1: 
            self.max_pooling_layer = layers.MaxPooling2D(pool_size=2, strides=self.stride, padding='same')

            num_heads = self.max_pooling_layer.compute_output_shape(input_shape)[-1] // self.head_dimension
            assert num_heads != 0
        else:
            num_heads = input_shape[-1] // self.head_dimension
        
        self.attention_layer = MHSAWithMultiHeadRelativePositionEmbedding(
            num_heads=num_heads, key_dim=self.head_dimension, out_shape=self.output_channel
        )

    def call(self, inputs):
        x = self.preact_layer_norm(inputs)
        if self.stride != 1:
            x = self.max_pooling_layer(x)
        return self.attention_layer(x)

class FFNBlock(layers.Layer):
    def __init__(self, expansion=4, kernel_size=1, activation='gelu'):
        super().__init__()
        
        self.expansion = expansion
        self.activation = activation
        self.kernel_size = kernel_size

    def build(self, input_shape):
        input_channel = input_shape[-1] 

        self.preact_layer_norm = layers.LayerNormalization(epsilon=1e-5)
        self.conv1 = Conv2DNoBias(input_channel * self.expansion, kernel_size=self.kernel_size)
        self.act = layers.Activation(self.activation)
        self.conv2 = Conv2DNoBias(input_channel, kernel_size=self.kernel_size)

    def call(self, inputs):
        preact = self.preact_layer_norm(inputs)
        x = self.conv1(preact)
        x = self.act(x)
        return self.conv2(x)

class STEMBlock(layers.Layer):
    def __init__(self, filters, activation, strides=2 ,use_bias=False):
        super().__init__()
        self.convbn1 = ConvBN2D(filters=filters, kernel_size=3, strides=strides, padding="same", activation=activation, use_bias=use_bias)
        self.convbn2 = Conv2DNoBias(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=use_bias)

    def call(self, inputs):
        x = self.convbn1(inputs)
        return self.convbn2(x)
 
class ResidualBlock(layers.Layer):
    def __init__(self, sub_layer, output_channel, stride, conv_shortcut=True, drop_rate=0, sthotastic_depth_rate=0):
        super().__init__()
        stride = stride[0] if isinstance(stride, tuple) else stride

        self.sthotastic_depth_rate = sthotastic_depth_rate
        self.sub_layer = sub_layer
        self.output_channel = output_channel
        self.stride = stride
        self.drop_rate = drop_rate
        self.conv_shortcut = conv_shortcut

        if conv_shortcut:
            self.pooling_layer = layers.MaxPooling2D(stride, strides=stride, padding='same')
            self.conv_layer = Conv2DNoBias(output_channel, 1, 1)

        if self.drop_rate > 0: self.dropout_layer = layers.Dropout(rate=drop_rate)

        if self.sthotastic_depth_rate > 0: self.std_layer = layers.Lambda(lambda x: layers.Add()([x[0],x[1] * sthotastic_depth_rate]))
        else: self.add_layer = layers.Add()

    def call(self, inputs):
        output = self.sub_layer(inputs)

        if self.conv_shortcut:
            resiudal = self.pooling_layer(inputs) if self.stride > 1 else inputs
            resiudal = self.conv_layer(resiudal)
        else:
            resiudal = inputs

        if self.drop_rate > 0: output = self.dropout_layer(output)

        if self.sthotastic_depth_rate > 0:
            return self.std_layer([output, resiudal])
        else: 
            return self.add_layer([output, resiudal])

class MBConvBlock(layers.Layer):
    def __init__(self, expansion_ratio, output_channel, stride, se_ratio=None, use_dw_strides=True, activation='gelu'):
        super(MBConvBlock, self).__init__()
        self.expansion_ratio = expansion_ratio
        self.output_channel = output_channel
        self.se_ratio = se_ratio
        self.activation = activation
        
        self.dw_strides, self.conv_strides = (stride, 1) if use_dw_strides else (1, stride)

    def build(self, input_shape):
        self.preact_batch_norm = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        
        exp_dim = int(input_shape[-1] * self.expansion_ratio)

        self.expand_convbn = ConvBN2D(exp_dim, kernel_size=1, strides=self.conv_strides, padding='same', activation=self.activation) #gelu? ve strides depthwisede 2 değil burada 2

        self.dw_conv = DepthwiseConv2DNoBias(kernel_size=3, strides=self.dw_strides, padding='same')
        self.dw_bn = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.dw_act = layers.Activation(self.activation)

        if self.se_ratio:
            self.se_block = SEBlock(se_ratio=self.se_ratio/self.expansion_ratio, activation=self.activation)

        self.project_conv = Conv2DNoBias(self.output_channel, 1, 1, padding='same')

    def call(self, inputs):
        preact = self.preact_batch_norm(inputs)
        x = self.expand_convbn(preact)
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        if self.se_ratio:
            x = self.se_block(x)
        x = self.project_conv(x)
        return x

class OutputBlock(layers.Layer):
    def __init__(self, num_classes, drop_rate, activation):
        super().__init__()
        self.drop_rate = drop_rate
        self.avg_pool = layers.GlobalAveragePooling2D()
        if drop_rate > 0:
            self.dropout_layer = layers.Dropout(drop_rate)
        self.dense_layer = layers.Dense(num_classes, activation=activation)
    
    def call(self, inputs):
        x = self.avg_pool(inputs)
        if self.drop_rate > 0:
            x = self.dropout_layer(x)
        return self.dense_layer(x)
