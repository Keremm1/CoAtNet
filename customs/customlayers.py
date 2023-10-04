from keras import layers
import tensorflow as tf
import numpy as np

class MultiHeadRelativePositionalEmbedding(layers.Layer):
    def __init__(self, with_cls_token=True, attn_height=-1, num_heads=-1, **kwargs):
        super().__init__(**kwargs)
        self.with_cls_token, self.attn_height, self.num_heads = with_cls_token, attn_height, num_heads
        if with_cls_token:
            self.cls_token_len = 1
            self.cls_token_pos_len = 3
        else:
            self.cls_token_len = 0
            self.cls_token_pos_len = 0

    def build(self, attn_shape):
        # input (with_cls_token=True): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width + class_token`
        # input (with_cls_token=False): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width`
        if self.attn_height == -1:
            height = width = int(float(attn_shape[2] - self.cls_token_len) ** 0.5)  # hh == ww, e.g. 14
        else:
            height = self.attn_height
            width = int(float(attn_shape[2] - self.cls_token_len) / height)
        num_heads = attn_shape[1] if self.num_heads == -1 else self.num_heads
        num_relative_distance = (2 * height - 1) * (2 * width - 1) + self.cls_token_pos_len
        # pos_shape = (num_relative_distance, num_heads)
        pos_shape = (num_heads, num_relative_distance)
        # initializer = tf.random_normal_initializer(stddev=0.02)
        self.relative_position_bias_table = self.add_weight(name="positional_embedding", shape=pos_shape, initializer="zeros", trainable=True)

        hh, ww = np.meshgrid(range(height), range(width))  # tf.meshgrid is same with np.meshgrid 'xy' mode, while torch.meshgrid 'ij' mode
        coords = np.stack([hh, ww], axis=-1)  # [14, 14, 2]
        coords_flatten = np.reshape(coords, [-1, 2])  # [196, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [196, 196, 2]
        relative_coords_hh = relative_coords[:, :, 0] + height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + width - 1) * (2 * height - 1)
        relative_coords = np.stack([relative_coords_hh, relative_coords_ww], axis=-1)

        relative_position_index = np.sum(relative_coords, axis=-1).astype("float32")  # [196, 196]
        if attn_shape[3] != attn_shape[2]:
            # Choose the small values if value_block != query_block
            relative_position_index = relative_position_index[:, -(attn_shape[3] - self.cls_token_len) :]

        if self.with_cls_token:
            top = np.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (num_relative_distance - 3)
            left = np.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (num_relative_distance - 2)
            corner = np.ones((1, 1), dtype=relative_position_index.dtype) * (num_relative_distance - 1)
            # print(f">>>> {top.shape = }, {left.shape = }, {corner.shape = }")
            # >>>> top.shape = TensorShape([1, 196]), left.shape = TensorShape([196, 1]), corner.shape = TensorShape([1, 1])
            left_corner = np.concatenate([corner, left], axis=0)
            relative_position_index = np.concatenate([top, relative_position_index], axis=0)
            relative_position_index = np.concatenate([left_corner, relative_position_index], axis=1)  # [197, 197]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("relative_position_index", tf.convert_to_tensor(relative_position_index, dtype="int64"), persistent=False)
        else:
            self.relative_position_index = tf.convert_to_tensor(relative_position_index, dtype="int64")
        super().build(attn_shape)

    def call(self, inputs, **kwargs):
        pos_emb = tf.gather(self.relative_position_bias_table, self.relative_position_index[: inputs.shape[2], : inputs.shape[3]], axis=1)
        # tf.print(pos_emb.shape, inputs.shape)
        return inputs + pos_emb




class MHSAWithMultiHeadRelativePositionEmbedding(layers.Layer):
    def __init__(
        self,
        num_heads=4,
        key_dim=0,
        global_query=None,
        out_shape=None,
        out_weight=True,
        qkv_bias=False,
        out_bias=False,
        attn_dropout=0,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.global_query = global_query
        self.out_shape = out_shape
        self.out_weight = out_weight
        self.qkv_bias = qkv_bias
        self.out_bias = out_bias
        self.attn_dropout = attn_dropout

    def build(self, input_shape):
        self.input_channel = input_shape[-1]
        self.height, self.width = input_shape[1:-1]

        self.key_dim = self.key_dim if self.key_dim > 0 else self.input_channel // self.num_heads
        self.out_shape = self.input_channel if self.out_shape is None or not self.out_weight else self.out_shape
        self.qk_out = self.num_heads * self.key_dim
        self.vv_dim = self.key_dim
        self.blocks = self.height * self.width
            
        if self.global_query: self.conv_layer = Conv2DNoBias(self.qk_out * 2, kernel_size=1, use_bias=self.qkv_bias)
        else: self.conv_layer = Conv2DNoBias(self.qk_out * 3, kernel_size=1, use_bias=self.qkv_bias)

        self.pos_emb = MultiHeadRelativePositionalEmbedding(with_cls_token=False, attn_height=self.height)

        output_dim = self.out_shape if self.out_shape > 0 else self.key_dim #girilen değerin şekli korunur
        self.output_dense = layers.Dense( units=output_dim, use_bias=self.out_bias)

    def call(self, inputs):
        #inputs ile num_heads ve key_dime bağlı ağırlıklarla çarpılır ve qvk değerleri çıkarılır. 
        if self.global_query:
            kv = self.conv_layer(inputs)
            kv = tf.reshape(kv, [-1, self.blocks, kv.shape[-1]])
            key, value = tf.split(kv, [self.qk_out, self.out_shape], axis=-1)
            query = self.global_query
            _, key, value = self._qkv_to_multi_head_channels_last_format(None, key, value, self.num_heads)
        else:
            qkv = self.conv_layer(inputs)
            query, key, value = tf.split(qkv, [self.qk_out, self.qk_out, self.qk_out], axis=-1)
            query, key, value = self._qkv_to_multi_head_channels_last_format(query, key, value, self.num_heads)
        output_shape = (self.height, self.width, self.out_shape)
        out = self._scaled_dot_product_attention(query, key, value, output_shape, self.pos_emb, out_weight=self.out_weight, dropout=self.attn_dropout)
        return out

    def _scaled_dot_product_attention(self, query, key, value, output_shape, pos_emb=None, out_weight=True, qk_scale=-1, dropout=0, name=None):
        blocks = output_shape[1:-1] if output_shape[0] is None or output_shape[0] < 1 else output_shape[:-1] #batch belirtilmediyse output_shape Batch size kapsamasın
        # query, value: [batch, num_heads, blocks, key_dim], key: [batch, num_heads, key_dim, blocks]
        qk_scale = qk_scale if qk_scale > 0 else (1.0 / (float(query.shape[-1]) ** 0.5)) #q.k'nin bölüneceği kök değeri
        attention_scores = query @ key #q.k (matrix multiplication) matrix çarpımı sayesinde her veri biribiriyle teker teker diğer verilerle çarpılabilir
        if qk_scale != 1:
            attention_scores = attention_scores * qk_scale #q.k / kök(q)
        if pos_emb is not None:
            attention_scores = pos_emb(attention_scores) #relative positional embedding
        attention_scores = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores) #q.k^T değerlerinin ölçekleri
        if dropout > 0:
            attention_scores = layers.Dropout(dropout, name=name and name + "attn_drop")(attention_scores)
        attention_output = attention_scores @ value #ilişki skoru
        output = tf.transpose(attention_output, [0, 2, 1, 3])  # [batch, q_blocks, num_heads, key_dim * attn_ratio]
        output = layers.Reshape([*blocks, int(np.prod(output.shape[2:]))])(output) if -1 in blocks else layers.Reshape([*blocks, -1])(output) # [B, hh, ww, out_shape(input_channel veya belirtildiyse out_channel)]
        #output için qkv değerleri ile bir [batch, q_blocks, num_heads, key_dim * attn_ratio] çıkarıldı baştaki hhi qq şekli üstüne bu değerler eklendi
        if out_weight:
            # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
            output = self.output_dense(output)
        return output

    @staticmethod
    def _qkv_to_multi_head_channels_last_format(query, key, value, num_heads):
        # print(f">>>> {query.shape = }, {key.shape = }, {value.shape = }, {num_heads = }")
        # resim verisindeki her piksel birbiriyle attention bazında işleme sokuluyor
        if query is not None:
            # query [batch, hh, ww, channel] -> [batch, num_heads, hh * ww, key_dim]
            query = layers.Reshape([-1, num_heads, query.shape[-1] // num_heads])(query)
            query = tf.transpose(query, [0, 2, 1, 3])
        if key is not None:
            # key [batch, hh, ww, channel] -> [batch, num_heads, key_dim, hh * ww] #matrix multiplaction için key_dim farklı yere taşınır uygun hale getirilir
            key = layers.Reshape([-1, num_heads, key.shape[-1] // num_heads])(key)
            key = tf.transpose(key, [0, 2, 3, 1])
        if value is not None: 
            # value [batch, hh, ww, channel] -> [batch, num_heads, hh * ww, vv_dim]
            value = layers.Reshape([-1, num_heads, value.shape[-1] // num_heads])(value)
            value = tf.transpose(value, [0, 2, 1, 3])
        # print(f">>>> {query.shape = }, {key.shape = }, {value.shape = }, {num_heads = }")
        return query, key, value


class Conv2DNoBias(layers.Layer):
    def __init__(self, filters, kernel_size=1, strides=1, padding="valid", use_bias=False, groups=1, use_torch_padding=True):
        # Initialize the Conv2D layer with use_bias set to False
        super().__init__()      

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)

        if isinstance(padding, str):
            padding = padding.lower()
            pad = (kernel_size[0] // 2, kernel_size[1] // 2) if use_torch_padding and padding == "same" else (0, 0)
        else:
            pad = padding if isinstance(padding, (list, tuple)) else (padding, padding)
            padding = "same" if max(pad) > 0 else "valid"
        
        if use_torch_padding and padding == "same":
            self.pad_layer = layers.ZeroPadding2D(padding=pad)
            padding = "valid"

        self.padding = padding
        self.use_torch_padding = use_torch_padding
        self.pad = pad
        
        groups = max(1, groups)

        self.conv = layers.Conv2D(filters,
            kernel_size,
            strides,
            "valid" if padding == "valid" else (pad if use_torch_padding else "same"),
            use_bias=use_bias,
            groups=groups,
        )
    
    def call(self, inputs):
        if self.use_torch_padding and self.padding == "valid":
            inputs = self.pad_layer(inputs) if max(self.pad) > 0 else inputs
        return self.conv(inputs)
    


class DepthwiseConv2DNoBias(layers.Layer):
    def __init__(self, kernel_size=1, strides=1, padding="valid", use_bias=False, use_torch_padding=True):
        # use_bias parametresinin False olmasını zorlayan normal bir Conv2D katmanı, aynı zamanda Padding'i manuel olarak uygular
        super().__init__()      
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size)

        if isinstance(padding, str):
            padding = padding.lower()
            pad = (kernel_size[0] // 2, kernel_size[1] // 2) if use_torch_padding and padding == "same" else (0, 0)
        else:
            pad = padding if isinstance(padding, (list, tuple)) else (padding, padding)
            padding = "same" if max(pad) > 0 else "valid"
        
        if use_torch_padding and padding == "same":
            self.pad_layer = layers.ZeroPadding2D(padding=pad)
            padding = "valid"

        self.padding = padding
        self.use_torch_padding = use_torch_padding
        self.pad = pad
        
        self.conv = layers.DepthwiseConv2D(
            kernel_size,
            strides,
            "valid" if padding == "valid" else (pad if use_torch_padding else "same"),
            use_bias=use_bias,
        )
    
    def call(self, inputs):
        if self.use_torch_padding and self.padding == "valid":
            inputs = self.pad_layer(inputs) if max(self.pad) > 0 else inputs
        return self.conv(inputs)

class ConvBN2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None, use_bias=False):
        super().__init__()
        self.conv = Conv2DNoBias(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)
        self.bn = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.act = layers.Activation(activation)
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)      
        return self.act(x) 