import tensorflow as tf


tf.random.set_seed(1024)
print(tf.config.list_physical_devices('GPU'))
tf.config.LogicalDeviceConfiguration(memory_limit=1024*12)

from attention_models.model_utils import return_best_model


from data import load_cifar100
(input_shape, num_classes), train_set, val_set = load_cifar100()



search_epochs = 15
model = return_best_model(input_shape, num_classes, train_set, val_set, search_epochs)