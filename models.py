from keras import Model, layers
from customblocks import ResidualBlock, STEMBlock, FFNBlock, MBConvBlock, MSHABlock, OutputBlock

class CoAtNet(Model):
    def __init__(self, 
                 num_classes, 
                 num_blocks=[2, 3, 5, 2],
                 out_channels=[96, 192, 384, 768],
                 expansion_rate=4, 
                 se_ratio=0.25,
                 sthotastic_depth_rate=0,
                 drop_connect_rate=0,
                 drop_rate=0,
                 block_types=["convulution","convulution","transformer","transformer"],
                 strides=[2,2,2,2],
                 head_dimension=32,
                 stem_filters=64,
                 stem_strides=2,
                 activation='gelu',
                 classifier_activation = 'softmax',
                 use_dw_strides=True
                 ):
                      
        super().__init__()
        
        assert len(num_blocks) == len(out_channels) == len(block_types) == len(strides)        
        
        self.stem_block = STEMBlock(stem_filters, activation=activation, strides=stem_strides)
        
        global_block_id = 0
        self.blocks = []
        total_blocks = sum(num_blocks)
        for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
            stack_stride = strides[stack_id]
            is_conv_block = True if block_type[0].lower() == "c" else False
            if is_conv_block: stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, list) else se_ratio
            for block_id in range(num_block):
                block_stride = stack_stride if block_id == 0 else 1
                block_conv_short_cut = True if block_id == 0 else False
                if is_conv_block: block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, list) else stack_se_ratio
                block_drop_rate = drop_connect_rate * global_block_id / total_blocks
                global_block_id += 1

                if is_conv_block:
                    blocks = (MBConvBlock(
                        expansion_rate, out_channel, block_stride, block_se_ratio, use_dw_strides, activation=activation
                        ),)
                else:
                    blocks = (
                        MSHABlock(
                            out_channel, block_stride, head_dimension
                        ),
                        FFNBlock(
                            expansion_rate, activation=activation
                        )
                    )
                    
                for block in blocks:                       
                    self.blocks.append(ResidualBlock(
                                block, out_channel, block_stride, False if isinstance(block, FFNBlock) else block_conv_short_cut, 
                                block_drop_rate, sthotastic_depth_rate
                    ))

        self.classification_output_block = OutputBlock(num_classes, drop_rate, classifier_activation)

    def call(self, inputs):
            x = self.stem_block(inputs)

            for block in self.blocks:
                x = block(x)

            return self.classification_output_block(x)


from tensorflow_addons.optimizers import AdamW
from keras import losses, metrics
from keras_tuner import HyperModel, HyperParameters
from functools import reduce

class CoAtNetHyperModel(HyperModel):
    def __init__(self, num_classes):
        super().__init__()
    
        self.num_classes = num_classes

    def build(self, hp :HyperParameters):
        # conv_num_blocks = hp.Int("conv_num_blocks", 1, 2, 1)
        # transformer_num_blocks = hp.Int("conv_num_blocks", 1, 2, 1)
        
        # strides = []
        # num_blocks = []
        # block_types = []

        # for num in range(conv_num_blocks):
        #     num_blocks.append(hp.Int(f"conv{num}_block_num", 1, 2, 1) if num == 0 else hp.Int(f"conv{num}_block_num", num_blocks[-1]+1, 6, 1))
        #     strides.append(2)
        #     block_types.append('C')
        
        # for num in range(transformer_num_blocks):
        #     num_blocks.append(hp.Int(f"transformer{num}_block_num", 1, 2, 1)) if num == 0 else num_blocks.insert(len(num_blocks)-1, hp.Int(f"transformer{num}_block_num", num_blocks[-1]+1, 7, 1))
        #     strides.append(2)
        #     block_types.append('T')


        # out_channels = [hp.Int(f"conv{0}_block_out_channel", 16, 96, 16)]
        # for _ in range(3):
        #     out_channels.append(out_channels[-1] * 2)
        
        # print(num_blocks, out_channels, block_types, strides)
        model = CoAtNet(num_classes=self.num_classes, 
                         expansion_rate=3,
                         se_ratio=0, 
        #                 drop_connect_rate=hp.Float("drop_connect_rate", 0, 0.99, 0.05), 
        #                 drop_rate=hp.Float("drop_rate", 0, 0.99, 0.05),
                         num_blocks=[2, 7, 14, 2], 
                        #  out_channels=[64, 128, 256, 512], 
        #                 block_types=block_types, 
        #                strides=strides, 
                         stem_strides=1, 
        #                 head_dimension=hp.Int("head_dimension", 16, out_channels[1], 16), 
                         stem_filters=48, 
                    )
    
        model.compile(
            optimizer=AdamW(
                learning_rate= 0.0010478, #0.0010478, manuel1e-4
                weight_decay= 0.00024111430262138014),
            loss=losses.CategoricalCrossentropy(), #from_logits, label_smoothing
            metrics=[
                    metrics.CategoricalAccuracy(name="accuracy"),
                    metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ])
    
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        batch_size=80
        return model.fit(
            *args,
            batch_size=batch_size,
            **kwargs,
        )