import tensorflow as tf


def ShiftUnit(inp, inp_ch, max_dil):
    def conv2d(inp, fltrs, k_size=(1, 1), dil_rate=(1, 1), act="relu"):
        x = tf.keras.layers.Conv2D(
            fltrs,
            kernel_size=k_size,
            strides=(1, 1),
            padding="same",
            dilation_rate=dil_rate,
            activation=act,
        )(inp)
        return tf.keras.layers.BatchNormalization()(x)

    def dwconv2d_pool(inp, max_dil):
        outs = []
        for i in range(1, max_dil + 1):
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                dilation_rate=(i, i),
                padding="same",
                activation="relu",
            )(inp)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            outs.append(x)
        return tf.keras.layers.Concatenate(axis=-1)(outs)

    x = conv2d(inp, inp_ch * 4)
    x = dwconv2d_pool(x, max_dil)
    x = conv2d(x, inp_ch * 2)

    return x
