import tensorflow as tf
import numpy as np





################################################
def repeat(tensor, repeats, axis=0, name="repeat"):
    """Repeats a `tf.Tensor`'s elements along an axis by custom amounts.

    Equivalent to Numpy's `np.repeat`.
    `tensor and `repeats` must have the same numbers of elements along `axis`.

    Args:
      tensor: A `tf.Tensor` to repeat.
      repeats: A 1D sequence of the number of repeats per element.
      axis: An axis to repeat along. Defaults to 0.
      name: (string, optional) A name for the operation.

    Returns:
      The `tf.Tensor` with repeated values.
    """
    with tf.name_scope(name):
        cumsum = tf.cumsum(repeats)
        range_ = tf.range(cumsum[-1])

        indicator_matrix = tf.cast(tf.expand_dims(range_, 1) >= cumsum, tf.int32)
        indices = tf.reduce_sum(indicator_matrix, reduction_indices=1)

        shifted_tensor = _axis_to_inside(tensor, axis)
        repeated_shifted_tensor = tf.gather(shifted_tensor, indices)
        repeated_tensor = _inside_to_axis(repeated_shifted_tensor, axis)

        shape = tensor.shape.as_list()
        shape[axis] = None
        repeated_tensor.set_shape(shape)

        return repeated_tensor

def _axis_to_inside(tensor, axis):
        """Shifts a given axis of a tensor to be the innermost axis.

        Args:
          tensor: A `tf.Tensor` to shift.
          axis: An `int` or `tf.Tensor` that indicates which axis to shift.

        Returns:
          The shifted tensor.
        """

        axis = tf.convert_to_tensor(axis)
        rank = tf.rank(tensor)

        range0 = tf.range(0, limit=axis)
        range1 = tf.range(tf.add(axis, 1), limit=rank)
        perm = tf.concat([[axis], range0, range1], 0)

        return tf.transpose(tensor, perm=perm)

def _inside_to_axis(tensor, axis):
        """Shifts the innermost axis of a tensor to some other axis.

        Args:
          tensor: A `tf.Tensor` to shift.
          axis: An `int` or `tf.Tensor` that indicates which axis to shift.

        Returns:
          The shifted tensor.
        """

        axis = tf.convert_to_tensor(axis)
        rank = tf.rank(tensor)

        range0 = tf.range(1, limit=axis + 1)
        range1 = tf.range(tf.add(axis, 1), limit=rank)
        perm = tf.concat([range0, [0], range1], 0)

        return tf.transpose(tensor, perm=perm)


##############################################################################
def spp_layer(input_, levels=4, name='SPP_layer', pool_type='max_pool'):
    '''
    Multiple Level SPP layer.

    Works for levels=[1, 2, 3, 6].
    '''

    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):

        for l in range(levels):

            l = l + 1
            ksize = [1, np.ceil(shape[1] / l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]

            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]

            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1), )

            else:
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool, (shape[0], -1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1:

                x_flatten = tf.reshape(pool, (shape[0], -1))
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))

    return x_flatten