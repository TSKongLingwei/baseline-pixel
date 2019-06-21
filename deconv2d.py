import tensorflow as tf

def deconv2d(x, kernel, strides, output_shape, activation=None):
    deconv = tf.layers.conv2d_transpose(inputs=x, filters=output_shape, kernel_size= kernel, strides=strides, padding="SAME")
    # kernel_shape = [kernel, kernel, 1, 1]
    # strides = [1, strides, strides, 1]
    # kernel = tf.get_variable('weight_{}'.format(name), shape=kernel_shape,
    #                          initializer=tf.random_normal_initializer(mean=0, stddev=1))
    # deconv = tf.nn.conv2d_transpose(x, kernel, strides=strides, output_shape=output_shape, padding='SAME',
    #                                 name='upsample_{}'.format(name))

    # # Now output.get_shape() is equal (?,?,?,?) which can become a problem in the
    # # next layers. This can be repaired by reshaping the tensor to its shape:
    # deconv = tf.reshape(deconv, output_shape)
    # # now the shape is back to (?, H, W, C) or (?, C, H, W)
    #
    # if training != False:
    #     deconv = tf.layers.batch_normalization(deconv, training=training, name='bn{}'.format(name))
    # if activation is None:
    #     return deconv
    #
    # deconv = activation(deconv, name='sigmoid_{}'.format(name))

    return deconv
