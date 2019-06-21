import tensorflow as tf

def up_sample_layer(input,shape,sample_factor):
    height=shape[1]
    width=shape[2]
    up_sample = tf.image.resize_bilinear(input,size=[height*sample_factor,width*sample_factor])
    #up_sample = tf.image.resize_images(input, [48, 48], method=0)
    #subpixel = tf.concat(
    #    [tf.transpose(
    #        tf.reshape(
    #            tf.transpose(
    #                tf.reshape(a, (-1, height, channel_num, sample_factor, sample_factor)), [0, 2, 1, 3, 4]), [-1, channel_num, height * sample_factor, sample_factor]), [0, 2, 3, 1]) for a
    #        in tf.split(input, width, 2)], 2)
    return up_sample
