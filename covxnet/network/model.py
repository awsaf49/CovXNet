import tensorflow as tf
from covxnet.blocks.residual import ResUnit
from covxnet.blocks.shifter import ShiftUnit
from covxnet.blocks.stem import Stem


# Configurations for each architecture
configs = {
    "CovXNet32": {"max_dil": 3, "depth": 5},
    "CovXNet64": {"max_dil": 4, "depth": 5},
    "CovXNet128": {"max_dil": 5, "depth": 5},
    "CovXNet256": {"max_dil": 6, "depth": 2},
}


def CovXNet(input_shape, num_classes, config):
    xin = tf.keras.layers.Input(shape=input_shape)
    x = Stem(xin)

    for max_dil in range(config["max_dil"], 1, -1):
        i = config["max_dil"] - max_dil
        inp_ch = 32*(2**i)
        depth = config["depth"]
        x = ResUnit(x, inp_ch=inp_ch, max_dil=max_dil, n_units=depth)
        if max_dil > 2:
            x = ShiftUnit(x, inp_ch=inp_ch, max_dil=max_dil)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    xout = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(xin, xout)
    return model


def CovXNet32(input_shape, num_classes):
    return CovXNet(input_shape, num_classes, configs["CovXNet32"])


def CovXNet64(input_shape, num_classes):
    return CovXNet(input_shape, num_classes, configs["CovXNet64"])


def CovXNet128(input_shape, num_classes):
    return CovXNet(input_shape, num_classes, configs["CovXNet128"])


def CovXNet256(input_shape, num_classes):
    return CovXNet(input_shape, num_classes, configs["CovXNet256"])
