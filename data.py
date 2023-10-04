from keras.datasets import cifar100, mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

def load_cifar100_wbatch(batch_size):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    input_shape = (32,32,3)
    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # and augmentation
    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)

    return (input_shape, num_classes),  datagen.flow(x_train, y_train, batch_size=batch_size), (x_test, y_test), x_train.shape[0] // batch_size

def load_cifar10_wbatch(batch_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    input_shape = (32,32,3)
    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # and augmentation
    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)

    return (input_shape, num_classes),  datagen.flow(x_train, y_train, batch_size=batch_size), (x_test, y_test), len(x_train) // batch_size


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    input_shape = (32,32,3)
    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (input_shape, num_classes), (x_train, y_train), (x_test, y_test)



def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = tf.expand_dims(x_train.astype('float32') / 255., -1)
    x_test = tf.expand_dims(x_test.astype('float32') / 255., -1)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


    input_shape = (28,28,1)
    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (input_shape, num_classes), (x_train, y_train), (x_test, y_test)


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train = tf.expand_dims(x_train.astype('float32') / 255., -1)
    x_test = tf.expand_dims(x_test.astype('float32') / 255., -1)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


    input_shape = (28,28,1)
    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (input_shape, num_classes), (x_train, y_train), (x_test, y_test)