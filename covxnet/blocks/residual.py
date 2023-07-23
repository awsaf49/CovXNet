import tensorflow as tf


def ResUnit(inp, inp_ch, max_dil, n_units):
    def unit_op(inp, fltrs, k_size=(1, 1), dil_rate=(1, 1), act="relu"):
        x = tf.keras.layers.Conv2D(
            fltrs,
            kernel_size=k_size,
            strides=(1, 1),
            padding="same",
            dilation_rate=dil_rate,
            activation=act,
        )(inp)
        return tf.keras.layers.BatchNormalization()(x)

    def depth_op(inp, max_dil, act="relu"):
        out_list = []
        for i in range(1, max_dil + 1):
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                dilation_rate=(i, i),
                padding="same",
                activation=act,
            )(inp)
            out_list.append(tf.keras.layers.BatchNormalization()(x))
        return tf.keras.layers.Concatenate(axis=-1)(out_list)

    for _ in range(n_units):
        x = unit_op(inp, inp_ch * 2)
        x = depth_op(x, max_dil)
        x = unit_op(x, inp_ch)
        inp = tf.keras.layers.Add()([x, inp])

    return x
