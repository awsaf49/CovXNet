import tensorflow as tf


def Stem(input):
    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )(input)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x
