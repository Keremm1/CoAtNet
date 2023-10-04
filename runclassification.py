import tensorflow as tf
from models import CoAtNet
from model_utils import evaluate_model
from tensorflow_addons.optimizers import AdamW
from keras import losses, metrics, optimizers
from tensorflow.keras.optimizers.schedules import CosineDecay
from customcallbacks import ValACCEarlyStopping

tf.random.set_seed(1024)
print(tf.config.list_physical_devices('GPU'))
tf.config.LogicalDeviceConfiguration(memory_limit=1024*12)

batch_size = 80
from data import load_cifar100, load_mnist, load_fashion_mnist, load_cifar100_wbatch, load_cifar10_wbatch
(input_shape, num_classes), train_set, val_set, steps_per_epoch = load_cifar100_wbatch(batch_size)

model= CoAtNet(num_classes=num_classes,
               expansion_rate=3,
               se_ratio=0,
                num_blocks=[2, 7, 14, 2], 
                stem_strides=1, 
                stem_filters=48, 
               )

# from keras_cv_attention_models.coatnet.coatnet import CoAtNet1
# model = CoAtNet1(input_shape, num_classes)

model.build((None, *input_shape))
model.summary()

model.compile(
            optimizer=AdamW(learning_rate=0.0010478, weight_decay=0.00024111430262138014),
            loss=losses.CategoricalCrossentropy(),
            metrics=[
                    metrics.CategoricalAccuracy(name="accuracy"),
                    metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ])

history = model.fit(
    train_set,
    epochs=50,
    validation_data=val_set,
    steps_per_epoch=steps_per_epoch,
    # callbacks=[ValACCEarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, verbose=1)]
)


evaluate_model(model, history, *val_set)