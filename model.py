"""
Derived from: https://github.com/ry/tensorflow-resnet
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from up_sample_layer import up_sample_layer
from deconv2d import deconv2d
import graph_nets as gn
import sonnet as snt
import util as ul

NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
CONV_WEIGHT_DECAY = 0.0000
CONV_WEIGHT_STDDEV = 0.01
BN_DECAY = 0.9
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_STDDEV = 0.01
train_able=True
CONV_WEIGHT_STDDEV_fc = 0.003
new_bn = True
class ResNetModel(object):

    def __init__(self, x, is_training, depth=50, num_classes=2):
        self.is_training = is_training
        self.num_classes = num_classes
        self.depth = depth
        if depth in NUM_BLOCKS:
            self.num_blocks = NUM_BLOCKS[depth]
        else:
            raise ValueError('Depth is not supported; it must be 50, 101 or 152')


    def inference(self, x):
        # Data BN
        with tf.variable_scope('data-bn'):
            x_bn = bn(x, is_training=False)

        s1 = self.scale1(x_bn)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)
        s4 = self.scale4(s3)
        s5 = self.scale5(s4)

        return s1,s2,s3, s4, s5


    def DGR(self, batch_x, batch_y=None, loss_weight=None):

        s1, s2, s3, s4, s5 = self.inference(batch_x)
        batch_y = tf.one_hot(batch_y, 2, axis=3)


        I5 =   conv1(s5, ksize=1, stride=1, filters_out=512)
        I4 =   conv1(s4, ksize=1, stride=1, filters_out=256)
        I3 =   conv1(s3, ksize=1, stride=1, filters_out=128)
        I2 =   conv1(s2, ksize=1, stride=1, filters_out=64)
        I1 =   conv1(s1, ksize=1, stride=1, filters_out=64)
        with tf.variable_scope('IFF5'):
            I4_5 =   conv1(I4, ksize=2, stride=2, filters_out=512)
            IFF5 = tf.concat([I4_5,I5],axis=3)
            IFF5 =   conv1(IFF5, ksize=1, stride=1, filters_out=512)
            IFF5 =   conv1(IFF5, ksize=3, stride=1, filters_out=512)

        with tf.variable_scope('IFF4'):

            I3_4 =   conv1(I3, ksize=2, stride=2, filters_out=256)
            I5_4 = deconv2d(x=I5, kernel=2, strides=2,output_shape=256)
            IFF4 = tf.concat([I3_4, I5_4,I4], axis=3)
            IFF4 =   conv1(IFF4, ksize=1, stride=1, filters_out=256)
            IFF4 =   conv1(IFF4, ksize=3, stride=1, filters_out=256)

        with tf.variable_scope('IFF3'):

            I2_3 =   conv1(I2, ksize=2, stride=2, filters_out=128)
            I4_3 = deconv2d(x=I4, kernel=2, strides=2, output_shape=128)
            IFF3 = tf.concat([I2_3, I4_3, I3], axis=3)
            IFF3 =   conv1(IFF3, ksize=1, stride=1, filters_out=128)
            IFF3 =   conv1(IFF3, ksize=3, stride=1, filters_out=128)
        with tf.variable_scope('IFF2'):
            I1_2 =   conv1(I1, ksize=2, stride=2, filters_out=64)
            I3_2 = deconv2d(x=I3, kernel=2, strides=2, output_shape=64)
            IFF2 = tf.concat([I1_2, I3_2, I2], axis=3)
            IFF2 =   conv1(IFF2, ksize=1, stride=1, filters_out=64)
            IFF2 =   conv1(IFF2, ksize=3, stride=1, filters_out=64)
        with tf.variable_scope('IFF1'):
            I2_1 = deconv2d(x=I2, kernel=2, strides=2, output_shape=64)
            IFF1 = tf.concat([I2_1, I1], axis=3)
            IFF1 =   conv1(IFF1, ksize=1, stride=1, filters_out=64)
            IFF1 =   conv1(IFF1, ksize=3, stride=1, filters_out=64)
        with tf.variable_scope('DGR'):
            map5 = deconv2d(x=IFF5, kernel=32, strides=32, output_shape=2)
            loss5 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=map5, labels=batch_y)
            map4 = deconv2d(x=IFF4, kernel=16, strides=16, output_shape=2)
            loss4 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=map4, labels=batch_y)
            fuse5_4 = tf.concat([map5,map4],axis=3)
            map3 = deconv2d(x=IFF3, kernel=8, strides=8, output_shape=2)
            map3 = tf.concat([fuse5_4,map3],axis=3)
            map3 =   conv1(map3, ksize=1, stride=1, filters_out=2, )
            loss3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=map3, labels=batch_y)

            map2 = deconv2d(x=IFF2, kernel=4, strides=4, output_shape=2)
            map2 = tf.concat([fuse5_4, map2], axis=3)
            map2 =   conv1(map2, ksize=1, stride=1, filters_out=2)
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=map2, labels=batch_y)

            fuse2_3 = tf.concat([map3,map2],axis=3)

            map1 = deconv2d(x=IFF1, kernel=2, strides=2, output_shape=2)
            map1 = tf.concat([fuse5_4, fuse2_3,map1], axis=3)
            map1 =   conv1(map1, ksize=1, stride=1, filters_out=2)
            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=map1, labels=batch_y)

            fuse_all = tf.concat([map1,map2,map3,map4,map5],axis=3)
            fuse_all=  conv1(fuse_all, ksize=3, stride=1, filters_out=64)
            fuse_all =   conv1(fuse_all, ksize=3, stride=1, filters_out=64)
            fuse_all =   conv1(fuse_all, ksize=3, stride=1, filters_out=2)
            lossfuse = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fuse_all, labels=batch_y)
        loss1_mean = tf.reduce_mean(loss1)
        loss2_mean = tf.reduce_mean(loss2)
        loss3_mean = tf.reduce_mean(loss3)
        loss4_mean = tf.reduce_mean(loss4)
        loss5_mean = tf.reduce_mean(loss5)
        lossfuse_mean = tf.reduce_mean(lossfuse)





        #score = tf.nn.softmax(f1_fuse, axis=3)
        score = tf.nn.softmax(fuse_all,axis=3)
        self.loss = loss1_mean + loss2_mean + loss3_mean + loss4_mean + loss5_mean+lossfuse_mean
        reg_loss_list = tf.losses.get_regularization_losses()
        reg_loss = 0.0
        if len(reg_loss_list) != 0:
            reg_loss = tf.add_n(reg_loss_list)
            self.loss += reg_loss

        lap_loss = tf.zeros(0)
        return self.loss, score, loss1_mean, lap_loss












    def loss(self, batch_x, batch_y=None, loss_weight=None):

        s1, s2, s3, s4, s5 = self.inference(batch_x)
        batch_y = tf.one_hot(batch_y, 2, axis=3)


        s5 = conv(s5, ksize=3, stride=1, filters_out=512 )

        s5 = deconv2d(x=s5, kernel=2, strides=2, name='upsample_5_map',
                      output_shape=512)

        f5_map = deconv2d(x=s5, kernel=16, strides=16, name='upsample_55',
                          output_shape=2)

        s5 = conv(s5, ksize=3, stride=1, filters_out=512  )

        loss5 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f5_map, labels=batch_y)
        loss5_mean = tf.reduce_mean(loss5)
        with tf.variable_scope('fuse5-4'):
            f4 = conv(s4, ksize=3, stride=1, filters_out=256  )

            f4 = tf.concat([s5, f4], axis=3, name='concat4-3')
            ################################################################################################################
            ################################################################################################################
            f4 = conv(f4, ksize=3, stride=1, filters_out=256  )

            f4_map = deconv2d(x=f4, kernel=16, strides=16, name='upsample_m4',
                              output_shape=2)

            f4 = deconv2d(x=f4, kernel=2, strides=2, name='upsample_f4',
                          output_shape=256)

            loss4 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f4_map, labels=batch_y)

            loss4_mean = tf.reduce_mean(loss4)
        with tf.variable_scope('fuse4-3'):
            f3 = conv(s3, ksize=3, stride=1, filters_out=128  )

            f3 = tf.concat([f4, f3], axis=3, name='concat4-3')
            ################################################################################################################
            ################################################################################################################
            f3 = conv(f3, ksize=3, stride=1, filters_out=128  )

            f3_map = deconv2d(x=f3, kernel=8, strides=8, name='upsample_m4',
                              output_shape=2)

            f3 = deconv2d(x=f3, kernel=2, strides=2, name='upsample_f3',
                          output_shape=128)

            loss3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f3_map, labels=batch_y)
            loss3_mean = tf.reduce_mean(loss3)

        with tf.variable_scope('fuse3-2'):
            f2 = conv(s2, ksize=3, stride=1, filters_out=64  )

            f2 = tf.concat([f3, f2], axis=3, name='concat3-2')
            ################################################################################################################
            ################################################################################################################
            f2 = conv(f2, ksize=3, stride=1, filters_out=64  )

            f2 = deconv2d(x=f2, kernel=2, strides=2, name='upsample_f2',
                          output_shape=64)




            f2_map = conv(f2, ksize=3, stride=1, filters_out=64  )
            f2_map = deconv2d(x=f2_map, kernel=2, strides=2, name='upsample_5_map',
                              output_shape=2)

            f2_fuse = conv(f2, ksize=3, stride=1, filters_out=64  )
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f2_map, labels=batch_y)
            loss2_mean = tf.reduce_mean(loss2)

        with tf.variable_scope('fuse2-1'):
            f1 = conv(s1, ksize=3, stride=1, filters_out=32  )
            f1 = tf.concat([f2_fuse, f1], axis=3, name='concat2-1')
            f1 = conv(f1, ksize=3, stride=1, filters_out=64  )

            f1 = deconv2d(x=f1, kernel=2, strides=2, name='upsample_f2',
                          output_shape=64)



            f1_fuse = conv(f1, ksize=3, stride=1, filters_out=64  )
            f1_fuse = conv(f1_fuse, ksize=3, stride=1, filters_out=64  )
            f1_fuse = conv(f1_fuse, ksize=3, stride=1, filters_out=2  )

            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f1_fuse, labels=batch_y)
            loss1_mean = tf.reduce_mean(loss1)


        score = tf.nn.softmax(f1_fuse, axis=3)
        # score = tf.nn.softmax(g6.nodes)
        self.loss = loss1_mean + loss2_mean + loss3_mean + loss4_mean + loss5_mean
        reg_loss_list = tf.losses.get_regularization_losses()
        reg_loss = 0.0
        if len(reg_loss_list) != 0:
            reg_loss = tf.add_n(reg_loss_list)
            self.loss += reg_loss

        lap_loss = tf.zeros(0)
        return self.loss, score, loss1_mean, lap_loss


    def loss_1(self, batch_x, batch_y=None, loss_weight=None):

        s1, s2, s3, s4, s5 = self.inference(batch_x)
        f5 = up_sample_layer(s5, s5.get_shape().as_list(), 2)
        f5g = conv(f5, ksize=3, stride=1, filters_out=32)
        f5g = tf.nn.relu(f5g)
        global_feature = ul.spp_layer(f5g, levels=3)

        conv6_s3, conv6_s4, conv6_s5 = self.res_sample(s3=s3, s4=s4, s5=s5, is_training=self.is_training,
                                                       name='upsample')
        conv6_s3 = tf.nn.relu(conv6_s3)
        conv6_s4 = tf.nn.relu(conv6_s4)
        conv6_s5 = tf.nn.relu(conv6_s5)

        conv6_s5 = deconv2d(x=conv6_s5, kernel=8, strides=4, name='upsample_s5',
                            output_shape=conv6_s3.get_shape().as_list()[3])
        conv6_s5 = tf.nn.relu(conv6_s5)
        # conv6_s5 = up_sample_layer(conv6_s5, conv6_s5.get_shape().as_list(), 4)
        conv6_s4 = deconv2d(x=conv6_s4, kernel=4, strides=2, name='upsample_s4',
                            output_shape=conv6_s3.get_shape().as_list()[3])
        conv6_s4 = tf.nn.relu(conv6_s4)
        # conv6_s4 = up_sample_layer(conv6_s4, conv6_s4.get_shape().as_list(), 2)
        with tf.variable_scope('conv_s34'):
            concat345 = tf.concat([conv6_s3, conv6_s4, conv6_s5], axis=3, name='concat345')
            ################################################################################################################
            ################################################################################################################
            conv345 = conv(concat345, ksize=3, stride=1, filters_out=512)
            conv345 = tf.nn.relu(conv345)
        ################################################################################################################
        ################################################################################################################
        with tf.variable_scope('conv7-net1'):
            conv7_2_net1 = conv(conv345, ksize=1, stride=1, filters_out=256)
            conv7_2_net1 = tf.nn.relu(conv7_2_net1)
        concat_up = deconv2d(x=conv7_2_net1, kernel=8, strides=8, name='upsample_concat',
                             output_shape=2)
        batch_y = tf.one_hot(batch_y, 2, axis=3)
        loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=concat_up, labels=batch_y)
        loss1_mean = tf.reduce_mean(loss1)

        score = tf.nn.softmax(concat_up, axis=3)
        # score = tf.nn.softmax(g6.nodes)
        self.loss = loss1_mean
        reg_loss_list = tf.losses.get_regularization_losses()
        reg_loss = 0.0
        if len(reg_loss_list) != 0:
            reg_loss = tf.add_n(reg_loss_list)
            self.loss += reg_loss

        lap_loss = tf.zeros(0)
        return self.loss, score, loss1_mean, lap_loss









    def load_original_weights(self, session, skip_layers=[]):
        A = tf.global_variables()
        weights_path = 'ResNet-L{}.npy'.format(self.depth)
        weights_dict = np.load(weights_path, encoding='bytes').item()
        for op_name in weights_dict:
            parts = op_name.split('/')

            if contains(op_name, skip_layers):
                print('fc')
                continue

            if parts[0] == 'fc' and self.num_classes != 1000:
                continue
            if 'weights' in parts:
                full_name = op_name[:-7] + 'conv2d/kernel:0'
            elif 'gamma' in parts:
                full_name = op_name[:-5] + 'bn/gamma:0'
            elif 'beta' in parts:
                full_name = op_name[:-4] + 'bn/beta:0'
            elif 'moving_variance' in parts:
                full_name = op_name[:-15] + 'bn/moving_variance:0'
            elif 'moving_mean' in parts:
                full_name = op_name[:-11] + 'bn/moving_mean:0'
            else:
                full_name = "{}:0".format(op_name)
            # A = tf.global_variables()
            var = [v for v in tf.global_variables() if v.name == full_name][0]
            session.run(var.assign(weights_dict[op_name]))





    def optimize(self, learning_rate, train_layers=[]):
        #trainable_var_names = ['weights', 'biases', 'beta', 'gamma']
        A = tf.trainable_variables()
        # var_list = [v for v in tf.trainable_variables() if
        #     v.name.split(':')[0].split('/')[-1] in trainable_var_names and
        #     contains(v.name, train_layers)]
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        return train_op




    def load_original_weights(self, session, skip_layers=[]):
        A = tf.global_variables()
        weights_path = 'ResNet-L{}.npy'.format(self.depth)
        weights_dict = np.load(weights_path, encoding='bytes').item()
        for op_name in weights_dict:
            parts = op_name.split('/')

            if contains(op_name, skip_layers):
                print('fc')
                continue

            if parts[0] == 'fc' and self.num_classes != 1000:
                continue
            if 'weights' in parts:
                full_name = op_name[:-7] + 'conv2d/kernel:0'
            elif 'gamma' in parts:
                full_name = op_name[:-5] + 'bn/gamma:0'
            elif 'beta' in parts:
                full_name = op_name[:-4] + 'bn/beta:0'
            elif 'moving_variance' in parts:
                full_name = op_name[:-15] + 'bn/moving_variance:0'
            elif 'moving_mean' in parts:
                full_name = op_name[:-11] + 'bn/moving_mean:0'
            else:
                full_name = "{}:0".format(op_name)
            # A = tf.global_variables()
            var = [v for v in tf.global_variables() if v.name == full_name][0]
            session.run(var.assign(weights_dict[op_name]))


    def scale1(self,input):
        # Scale 1
        with tf.variable_scope(name_or_scope='scale1'):
            s1_conv = conv(input, ksize=7, stride=2, filters_out=64, trainable=train_able)
            s1_bn = bn(s1_conv, is_training=self.is_training)
            s1 = tf.nn.relu(s1_bn)
        return s1

    def scale2(self, input):
        # Scale 2
        with tf.variable_scope(name_or_scope='scale2'):
            s2_mp = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            s2 = stack(s2_mp, is_training=self.is_training, num_blocks=self.num_blocks[0], stack_stride=1,
                       block_filters_internal=64, trainable=train_able)
        return s2

    def scale3(self, input):
        # Scale 3
        with tf.variable_scope(name_or_scope='scale3'):
            s3 = stack(input, is_training=self.is_training, num_blocks=self.num_blocks[1], stack_stride=2,
                       block_filters_internal=128, trainable=train_able)
        return s3

    def scale4(self, input):
        # Scale 4
        with tf.variable_scope(name_or_scope='scale4'):
            s4 = stack(input, is_training=self.is_training, num_blocks=self.num_blocks[2], stack_stride=2,
                       block_filters_internal=256, trainable=train_able)
        return s4

    def scale5(self, input):
        # Scale 5
        with tf.variable_scope(name_or_scope='scale5'):
            s5 = stack(input, is_training=self.is_training, num_blocks=self.num_blocks[3], stack_stride=2,
                       block_filters_internal=512, trainable=train_able)
        return s5

    def res_sample(self,s3, s4, s5, is_training, name):
        with tf.variable_scope(name):
            with tf.variable_scope('conv6_c3'):
                conv6_s3 = conv(s3, ksize=3, stride=1, filters_out=128)
            with tf.variable_scope('conv6_c4'):
                conv6_s4 = conv(s4, ksize=3, stride=1, filters_out=256)
            with tf.variable_scope('conv6_c5'):
                conv6_s5 = conv(s5, ksize=3, stride=1, filters_out=512)
            return conv6_s3, conv6_s4, conv6_s5





    def stage2_fuse(self, batch_x, label, segments, segments2,n_edge, receivers, senders, edge, n_node, base):

        s1, s2, s3, s4, s5 = self.inference(batch_x)
        batch_y = tf.one_hot(label, 2, axis=3)
        s5_g = conv(s5, ksize=1, stride=1, filters_out=32)
        global_feature = ul.spp_layer(s5_g, levels=3)

        s5 = conv(s5, ksize=3, stride=1, filters_out=512)
        s5 = tf.nn.relu(s5)

        s5 = deconv2d(x=s5, kernel=2, strides=2, name='upsample_5_map',
                                output_shape=512)

        f5_map = deconv2d(x=s5, kernel=16, strides=16, name='upsample_55',
                          output_shape=2)

        s5 = conv(s5, ksize=3, stride=1, filters_out=512)

        loss5 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f5_map, labels=batch_y)
        loss5_mean = tf.reduce_mean(loss5)
        with tf.variable_scope('fuse5-4'):
            f4 = conv(s4, ksize=3, stride=1, filters_out=256)

            f4 = tf.concat([s5, f4], axis=3, name='concat4-3')
            ################################################################################################################
            ################################################################################################################
            f4 = conv(f4, ksize=3, stride=1, filters_out=256)

            f4_map = deconv2d(x=f4, kernel=16, strides=16, name='upsample_m4',
                                output_shape=2)

            f4 = deconv2d(x=f4, kernel=2, strides=2, name='upsample_f4',
                                output_shape=256)

            loss4 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f4_map, labels=batch_y)

            loss4_mean = tf.reduce_mean(loss4)
        with tf.variable_scope('fuse4-3'):
            f3 = conv(s3, ksize=3, stride=1, filters_out=128)

            f3 = tf.concat([f4, f3], axis=3, name='concat4-3')
            ################################################################################################################
            ################################################################################################################
            f3 = conv(f3, ksize=3, stride=1, filters_out=128)

            f3_map = deconv2d(x=f3, kernel=8, strides=8, name='upsample_m4',
                              output_shape=2)


            f3 = deconv2d(x=f3, kernel=2, strides=2, name='upsample_f3',
                      output_shape=128)

            loss3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f3_map, labels=batch_y)
            loss3_mean = tf.reduce_mean(loss3)

        with tf.variable_scope('fuse3-2'):
            f2 = conv(s2, ksize=3, stride=1, filters_out=64)

            f2 = tf.concat([f3, f2], axis=3, name='concat3-2')
            ################################################################################################################
            ################################################################################################################
            f2 = conv(f2, ksize=3, stride=1, filters_out=64)

            f2 = deconv2d(x=f2, kernel=2, strides=2, name='upsample_f2',
                      output_shape=64)

            g2 = self.graph(segments2, f2, n_edge, receivers, senders, edge, n_node, global_feature)
            gw2_1 = self.gn_create([64], [64], [64], 'gw2_1')
            gw2_2 = self.gn_create([64], [64], [64], 'gw2_2')
            gw2_3 = self.gn_create([32], [32], [64], 'gw2_3')
            gw2_4 = self.gn_create([32], [1], [1], 'gw2_4')
            g2 = gw2_4(gw2_3(gw2_2(gw2_1(g2))))
            map2 = self.backtoimg(g2.nodes,segments2,base,g2.n_node,128)

            f2_fuse = tf.concat([f2, map2], axis=3, name='concat-f2-map2')
            f2_map =conv(f2_fuse, ksize=3, stride=1, filters_out=64)
            f2_map = deconv2d(x=f2_map, kernel=2, strides=2, name='upsample_5_map',
                          output_shape=2)




            f2_fuse = conv(f2_fuse, ksize=3, stride=1, filters_out=64)
            loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f2_map, labels=batch_y)
            loss2_mean = tf.reduce_mean(loss2)


        with tf.variable_scope('fuse2-1'):
            f1 = conv(s1, ksize=3, stride=1, filters_out=32)
            f1 = tf.concat([f2_fuse, f1], axis=3, name='concat2-1')
            f1 = conv(f1, ksize=3, stride=1, filters_out=64)

            f1 = deconv2d(x=f1, kernel=2, strides=2, name='upsample_f2',
                          output_shape=64)

            g1 = self.graph(segments, f1, n_edge, receivers, senders, edge, n_node, global_feature)
            gw1_1 = self.gn_create(64, 64, 64, 'gw1_1')
            gw1_2 = self.gn_create(64, 64, 64, 'gw1_2')
            gw1_3 = self.gn_create(32, 32, 64, 'gw1_3')
            gw1_4 = self.gn_create(32, 32, 1, 'gw1_4')
            g1 = gw1_4(gw1_3(gw1_2(gw1_1(g1))))
            map1 = self.backtoimg(g1.nodes, segments, base, g1.n_node, 256)
            f1_fuse = tf.concat([f1, map1], axis=3, name='concat-f1-map1')
            f1_fuse = conv(f1_fuse, ksize=3, stride=1, filters_out=64)
            f1_fuse = conv(f1_fuse, ksize=3, stride=1, filters_out=64)
            f1_fuse = conv(f1_fuse, ksize=3, stride=1, filters_out=2)

            loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=f1_fuse, labels=batch_y)
            loss1_mean = tf.reduce_mean(loss1)
     ################################################################################################################

        score = tf.nn.softmax(f1_fuse,axis=3)
        # score = tf.nn.softmax(g6.nodes)
        self.loss = loss1_mean+loss2_mean+loss3_mean+loss4_mean+loss5_mean
        reg_loss_list = tf.losses.get_regularization_losses()
        reg_loss = 0.0
        if len(reg_loss_list) != 0:
            reg_loss = tf.add_n(reg_loss_list)
            self.loss += reg_loss

        lap_loss = tf.zeros(0)

        return self.loss, score, lap_loss,loss1_mean,loss2_mean,loss3_mean,loss4_mean,loss5_mean





    def stage2_fuse_test(self, batch_x, label, segments, n_edge, receivers, senders, edge, n_node, base):

        s1, s2, s3, s4, s5 = self.inference(batch_x)
        f5 = up_sample_layer(s5, s5.get_shape().as_list(), 2)
        f5g = conv(f5, ksize=3, stride=1, filters_out=32)
        f5g = tf.nn.relu(f5g)
        global_feature = ul.spp_layer(f5g, levels=3)

        conv6_s3, conv6_s4, conv6_s5 = self.res_sample(s3=s3, s4=s4, s5=s5, is_training=self.is_training,
                                                       name='upsample')
        conv6_s3 = tf.nn.relu(conv6_s3)
        conv6_s4 = tf.nn.relu(conv6_s4)
        conv6_s5 = tf.nn.relu(conv6_s5)

        conv6_s2 = conv(s2, ksize=3, stride=1, filters_out=64)
        conv6_s1 = conv(s1, ksize=3, stride=1, filters_out=64)
        conv6_s2 = tf.nn.relu(conv6_s2)
        conv6_s1 = tf.nn.relu(conv6_s1)
        s2_up = deconv2d(x=conv6_s2, kernel=4, strides=4,name = "s2_up",
                            output_shape=64)
        s1_up = deconv2d(x=conv6_s1, kernel=2, strides=2,name = "s1_up" ,
                            output_shape=64)
#####################################################################
        conv6_s5 = deconv2d(x=conv6_s5, kernel=8, strides=4, name='upsample_s5',
                            output_shape=conv6_s3.get_shape().as_list()[3])
        conv6_s5 = tf.nn.relu(conv6_s5)
        # conv6_s5 = up_sample_layer(conv6_s5, conv6_s5.get_shape().as_list(), 4)
        conv6_s4 = deconv2d(x=conv6_s4, kernel=4, strides=2, name='upsample_s4',
                            output_shape=conv6_s3.get_shape().as_list()[3])
        conv6_s4 = tf.nn.relu(conv6_s4)
        # conv6_s4 = up_sample_layer(conv6_s4, conv6_s4.get_shape().as_list(), 2)
        with tf.variable_scope('conv_s34'):
            concat345 = tf.concat([conv6_s3, conv6_s4, conv6_s5], axis=3, name='concat345')
            ################################################################################################################
            ################################################################################################################
            conv345 = conv(concat345, ksize=3, stride=1, filters_out=512)
            conv345 = tf.nn.relu(conv345)
        ################################################################################################################
        ################################################################################################################
        with tf.variable_scope('conv7-net1'):
            conv7_2_net1 = conv(conv345, ksize=1, stride=1, filters_out=256)
            conv7_2_net1 = tf.nn.relu(conv7_2_net1)
        concat_up = deconv2d(x=conv7_2_net1, kernel=8, strides=8, name='upsample_concat',
                             output_shape=128)
        concat_all = tf.concat([s1_up,s2_up,concat_up],axis = 3)

        g1 = self.graph(segments, concat_up, n_edge, receivers, senders, edge, n_node, global_feature)
        graph_work0 = self.gn_create([128], [128], [128], 'gw2_1')
        graph_work1 = self.gn_create([128], [128], [128], 'gw2_1')
        graph_work2 = self.gn_create([64], [64], [128], 'gw2_2')
        graph_work3 = self.gn_create([64], [64], [128], 'gw2_2')
        graph_work4 = self.gn_create([64], [64], [128], 'gw2_3')
        graph_work5 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work6 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work7 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work8 = self.gn_create([128], [2], [2], 'gw2_4')

        g0 = graph_work0(g1)
        g2 = graph_work1(g0)
        g3 = graph_work2(g2)

        g4 = self.graph_add(graph_work3(g3), g3)
        g5 = self.graph_add(graph_work4(g4), g4)
        g6 = graph_work5(g5)
        # g5 = graph_work4(g4)

        g7 = self.graph_add(graph_work6(g6), g6)
        g8 = self.graph_add(graph_work7(g7), g7)
        gf = self.graph_concat(g2, g5)
        gf = self.graph_concat(gf, g8)
        gf = graph_work8(gf)

        #        return g1,g6

        node_label = tf.cast(tf.unsorted_segment_mean(label, segments, gf.n_node[0]),
                             dtype=tf.float32)
        label_mean = tf.reduce_mean(tf.cast(label, dtype=tf.float32))

        a = tf.ones_like(node_label, dtype=tf.int32)
        b = tf.zeros_like(node_label, dtype=tf.int32)
        condition = tf.less(label_mean, node_label)
        label_t = tf.where(condition, a, b)
        label_t = tf.squeeze(label_t, axis=1)
        label_t_onehot = tf.one_hot(label_t, 2, axis=1)
        loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gf.nodes, labels=label_t_onehot)
        loss1_mean = tf.reduce_mean(loss1)


        attention = tf.nn.softmax(gf.nodes)
        attention = attention[:,1]
        attention = attention[:,tf.newaxis]
        attention = tf.nn.softmax(attention,axis = 0)
        attention = self.backtoimg(attention, segments, base, g2.n_node, 256)

        # attention = tf.reshape(attention,[1,65536])
        # attention = tf.nn.softmax(attention)
        #attention = tf.reshape(attention,[1,256,256,1])
        attention = ul.repeat(attention,repeats = [concat_all.get_shape().as_list()[3]],axis =3 )

        final = tf.multiply(attention,concat_all)
        final = conv(final, ksize=3, stride=1, filters_out=64)
        final = tf.nn.relu(final)
        final = conv(final, ksize=3, stride=1, filters_out=64)
        final = tf.nn.relu(final)
        final = conv(final, ksize=3, stride=1, filters_out=2)
        label = tf.squeeze(label,axis = 3)
        batch_y = tf.one_hot(label, 2, axis=3)
        loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final, labels=batch_y)
        loss2_mean = tf.reduce_mean(loss2)
        score = tf.nn.softmax(final, axis=3)

        ################################################################################################################
        ################################################################################################################
        self.loss = loss1_mean+loss2_mean
        reg_loss_list = tf.losses.get_regularization_losses()
        reg_loss = 0.0
        if len(reg_loss_list) != 0:
            reg_loss = tf.add_n(reg_loss_list)
            self.loss += reg_loss
        # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss = tf.add_n([cross_entropy_mean] + [cross_entropy_mean_net2] + [cross_entropy_mean_net3] + regularization_losses)

        lap_loss = tf.zeros(0)

        return self.loss, loss1_mean, score, lap_loss,attention






    def stage2_loss(self, batch_x, label, segments, n_edge, receivers, senders, edge, n_node, global_fe,
                    use_graphonly=0):

        s1, s2, s3, s4, s5 = self.inference(batch_x)

        f5 = up_sample_layer(s5, s5.get_shape().as_list(), 2)
        f5g = conv(f5, ksize=3, stride=1, filters_out=32)
        global_feature = ul.spp_layer(f5g, levels=3)

        with tf.variable_scope('fuse5-4'):
            f4 = tf.concat([f5, s4], axis=3, name='concat5-4')
            ################################################################################################################
            ################################################################################################################
            f4 = conv(f4, ksize=3, stride=1, filters_out=512)

        f4 = up_sample_layer(f4, f4.get_shape().as_list(), 2)
        with tf.variable_scope('fuse4-3'):
            f3 = tf.concat([f4, s3], axis=3, name='concat4-3')
            ################################################################################################################
            ################################################################################################################
            f3 = conv(f3, ksize=3, stride=1, filters_out=256)

        f3 = up_sample_layer(f3, f3.get_shape().as_list(), 2)
        with tf.variable_scope('fuse3-2'):
            f2 = tf.concat([f3, s2], axis=3, name='concat3-2')
            ################################################################################################################
            ################################################################################################################
            f2 = conv(f2, ksize=3, stride=1, filters_out=128)

        f2 = up_sample_layer(f2, f2.get_shape().as_list(), 2)
        with tf.variable_scope('fuse2-1'):
            f1 = tf.concat([f2, s1], axis=3, name='concat2-1')
            ################################################################################################################
            ################################################################################################################
            f1 = conv(f1, ksize=3, stride=1, filters_out=64)
        fg = up_sample_layer(f1, f1.get_shape().as_list(), 2)


        g1 = self.graph(segments, fg, n_edge, receivers, senders, edge, n_node, global_feature)
        #
        # graph_work1 = gn.modules.GraphNetwork(
        #     edge_model_fn=lambda: snt.nets.MLP([64,64], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     node_model_fn=lambda: snt.nets.MLP([64,64], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     global_model_fn=lambda: snt.nets.MLP([128,128], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}))
        #
        # graph_work2 = gn.modules.GraphNetwork(
        #     edge_model_fn=lambda: snt.nets.MLP([64,64], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     node_model_fn=lambda: snt.nets.MLP([64,64], init20ializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     global_model_fn=lambda: snt.nets.MLP([128,128], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}))
        #
        # graph_work3 = gn.modules.GraphNetwork(
        #     edge_model_fn=lambda: snt.nets.MLP([32,32], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     node_model_fn=lambda: snt.nets.MLP([32,32], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     global_model_fn=lambda: snt.nets.MLP([64,64], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}))
        #
        # graph_work4 = gn.modules.GraphNetwork(
        #     edge_model_fn=lambda: snt.nets.MLP([32,32], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     node_model_fn=lambda: snt.nets.MLP([32,32], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     global_model_fn=lambda: snt.nets.MLP([64,64], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}))
        #
        # graph_work5 = gn.modules.GraphNetwork(
        #     edge_model_fn=lambda: snt.nets.MLP([32,32], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     node_model_fn=lambda: snt.nets.MLP([32,2], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
        #     global_model_fn=lambda: snt.nets.MLP([2], initializers={
        #         "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}))

        graph_work1 = self.gn_create([64], [64], [128], 'gw2_1')
        graph_work2 = self.gn_create([64], [64], [128], 'gw2_2')
        graph_work3 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work4 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work5 = self.gn_create([32], [2], [2], 'gw2_4')


        g2 = graph_work1(g1)
        g3 = graph_work2(g2)
        g4 = graph_work3(g3)
        g5 = graph_work4(g4)
        g6 = graph_work5(g5)

        #        return g1,g6
        node_label = tf.cast(tf.unsorted_segment_mean(label, segments, g6.n_node[0]),
                             dtype=tf.float32)
        label_mean = tf.reduce_mean(tf.cast(label, dtype=tf.float32))

        a = tf.ones_like(node_label, dtype=tf.int32)
        b = tf.zeros_like(node_label, dtype=tf.int32)
        condition = tf.less(label_mean, node_label)
        label_t = tf.where(condition, a, b)
        label_t = tf.squeeze(label_t, axis=1)
        label_t_onehot = tf.one_hot(label_t, 2, axis=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=g6.nodes, labels=label_t_onehot)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        score = tf.nn.softmax(g6.nodes)
        # L2范数

        # score = tf.sigmoid(g6.nodes)
        # MSE = tf.reduce_mean(tf.square(score - node_label))





        ################################################################################################################
        ################################################################################################################
        self.loss = cross_entropy_mean
        reg_loss_list = tf.losses.get_regularization_losses()
        reg_loss = 0.0
        if len(reg_loss_list) != 0:
            reg_loss = tf.add_n(reg_loss_list)
            self.loss += reg_loss
        # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss = tf.add_n([cross_entropy_mean] + [cross_entropy_mean_net2] + [cross_entropy_mean_net3] + regularization_losses)
        # tf.summary.scalar("CE-net1", cross_entropy_mean)
        # tf.summary.scalar("CE-net2", cross_entropy_mean_net2)
        # tf.summary.scalar("CE-net3", cross_entropy_mean_net3)
        # tf.summary.scalar("fisher-net1", fisherloss_net1)
        # tf.summary.scalar("fisher-net2", fisherloss_net2)
        # tf.summary.scalar("fisher-net3", fisherloss_net3)
        # tf.summary.scalar("reg-loss", reg_loss)
        # summary_op = tf.summary.merge_all()
        lap_loss = tf.zeros(0)

        return self.loss, cross_entropy_mean, score, label_t_onehot






    def stage2_loss_test(self, batch_x, label_t,e_label, segments, n_edge, receivers, senders, edge, n_node, global_fe,
                    use_graphonly=0):



        s1, s2, s3, s4, s5 = self.inference(batch_x)
        f5 = up_sample_layer(s5, s5.get_shape().as_list(), 2)
        f5g = conv(f5, ksize=3, stride=1, filters_out=32)
        f5g = tf.nn.relu(f5g)
        global_feature = ul.spp_layer(f5g, levels=3)

        conv6_s3, conv6_s4, conv6_s5 = self.res_sample(s3=s3, s4=s4, s5=s5, is_training=self.is_training,
                                                       name='upsample')
        conv6_s3 = tf.nn.relu(conv6_s3)
        conv6_s4 = tf.nn.relu(conv6_s4)
        conv6_s5 = tf.nn.relu(conv6_s5)

        conv6_s5 = deconv2d(x=conv6_s5, kernel=8, strides=4, name='upsample_s5',
                            output_shape=conv6_s3.get_shape().as_list()[3])
        conv6_s5 = tf.nn.relu(conv6_s5)
        # conv6_s5 = up_sample_layer(conv6_s5, conv6_s5.get_shape().as_list(), 4)
        conv6_s4 = deconv2d(x=conv6_s4, kernel=4, strides=2, name='upsample_s4',
                            output_shape=conv6_s3.get_shape().as_list()[3])
        conv6_s4 = tf.nn.relu(conv6_s4)
        # conv6_s4 = up_sample_layer(conv6_s4, conv6_s4.get_shape().as_list(), 2)
        with tf.variable_scope('conv_s34'):
            concat345 = tf.concat([conv6_s3, conv6_s4, conv6_s5], axis=3, name='concat345')
            ################################################################################################################
            ################################################################################################################
            conv345 = conv(concat345, ksize=3, stride=1, filters_out=512)
            conv345 = tf.nn.relu(conv345)
        ################################################################################################################
        ################################################################################################################
        with tf.variable_scope('conv7-net1'):
            conv7_2_net1 = conv(conv345, ksize=1, stride=1, filters_out=256)
            conv7_2_net1 = tf.nn.relu(conv7_2_net1)
        concat_up = deconv2d(x=conv7_2_net1, kernel=8, strides=8, name='upsample_concat',
                             output_shape=128)

        # x_bn = bn(batch_x, is_training=self.is_training)

        g1 = self.graph(segments, concat_up, n_edge, receivers, senders, edge, n_node, global_fe)


        # graph_work1 = self.gn_create([64], [64], [128], 'gw2_1')
        # graph_work2 = self.gn_create([64], [64], [128], 'gw2_2')
        # graph_work3 = self.gn_create([32], [32], [64], 'gw2_3')
        # graph_work4 = self.gn_create([32], [32], [64], 'gw2_3')
        # # graph_work5 = self.gn_create([32,32], [128,64,32,2], [2], 'gw2_4')
        # graph_work5 = self.gn_create([32], [2], [2], 'gw2_4')

        # graph_work1 = self.gn_create([64], [64], [128], 'gw2_1')
        # graph_work2 = self.gn_create([64], [64], [128], 'gw2_2')
        # graph_work3 = self.gn_create([64], [64], [128], 'gw2_2')
        # graph_work4 = self.gn_create([32], [32], [64], 'gw2_3')
        # graph_work5 = self.gn_create([32], [32], [64], 'gw2_3')
        # graph_work6 = self.gn_create([32], [32], [64], 'gw2_3')
        # graph_work7 = self.gn_create([32], [2], [2], 'gw2_4')

        graph_work0 = self.gn_create([128], [128], [128], 'gw2_1')
        graph_work1 = self.gn_create([128], [128], [128], 'gw2_1')
        graph_work2 = self.gn_create([64], [64], [128], 'gw2_2')
        graph_work3 = self.gn_create([64], [64], [128], 'gw2_2')
        graph_work4 = self.gn_create([64], [64], [128], 'gw2_3')
        graph_work5 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work6 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work7 = self.gn_create([32], [32], [64], 'gw2_3')
        graph_work8 = self.gn_create([3], [2], [2], 'gw2_4')

        g0 = graph_work0(g1)
        g2 = graph_work1(g0)
        g3 = graph_work2(g2)

        g4 = self.graph_add(graph_work3(g3),g3)
        g5 = self.graph_add(graph_work4(g4),g4)
        g6 = graph_work5(g5)
        # g5 = graph_work4(g4)

        g7 = self.graph_add(graph_work6(g6),g6)
        g8 = self.graph_add(graph_work7(g7),g7)
        gf = self.graph_concat(g2,g5)
        gf = self.graph_concat(gf,g8)
        gf = graph_work8(gf)

        #        return g1,g6
        e_label_onehot = tf.one_hot(e_label, 3, axis=1)
        loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gf.edges, labels=e_label_onehot)
        loss1_mean = tf.reduce_mean(loss1)

        label_t_onehot = tf.one_hot(label_t, 2, axis=1)
        loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gf.nodes, labels=label_t_onehot)
        loss2_mean = tf.reduce_mean(loss2)
        score1 = tf.nn.softmax(gf.nodes)
        score2 = tf.nn.softmax(gf.edges)


        ################################################################################################################
        ################################################################################################################
        self.loss = loss1_mean + loss2_mean
        reg_loss_list = tf.losses.get_regularization_losses()
        reg_loss = 0.0
        if len(reg_loss_list) != 0:
            reg_loss = tf.add_n(reg_loss_list)
            self.loss += reg_loss
        # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss = tf.add_n([cross_entropy_mean] + [cross_entropy_mean_net2] + [cross_entropy_mean_net3] + regularization_losses)

        lap_loss = tf.zeros(0)

        return self.loss, loss1_mean, loss2_mean,score1,score2




































    def backtoimg(self,node_feature,segments,base,n_node,length):
        num = tf.squeeze(n_node)
        base = base[:,0:length,0:length,0:num]
        base = tf.cast(base,tf.int32)
        segments = segments[:,:,:,tf.newaxis]
        segments = ul.repeat(segments,[num],axis=3)
        index = tf.equal(base,segments)
        index = tf.cast(index,tf.float32)
        node_feature = node_feature[tf.newaxis,tf.newaxis,:,:]
        out =   tf.nn.conv2d(index,filter=node_feature,strides=[1,1,1,1],padding = "SAME")
        return out


    def gn_create(self,edge,node,blonal_fn,name):
            x1= gn.modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP(edge, initializers={
                "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)} ),
            node_model_fn=lambda: snt.nets.MLP(node, initializers={
                "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}),
            global_model_fn=lambda: snt.nets.MLP(blonal_fn, initializers={
                "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)}))
            return  x1

    def gn_create1(self, edge, node, blonal_fn, name):
        x1 = gn.modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP(edge, initializers={
                "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)},
                                               regularizers=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY)),
            node_model_fn=lambda: snt.nets.MLP(node, initializers={
                "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)},
                                               regularizers=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY)),
            global_model_fn=lambda: snt.nets.MLP(blonal_fn, initializers={
                "w": tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV_fc)},
                                                 regularizers=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY)))
        return x1


    def graph(self,segments,features,n_edge,receivers,senders,edge,n_node,global_fe):

        node = tf.unsorted_segment_mean(features,segments,n_node[0])
        a = gn.graphs.GraphsTuple(
            n_edge = n_edge,
            n_node = n_node,
            nodes = node,
            edges = edge,
            receivers = receivers,
            senders = senders,
            globals =global_fe
         )
        return a


    def graph_concat(self,g1,g2):

        a = gn.graphs.GraphsTuple(
                n_edge=g1.n_edge,
                n_node=g1.n_node,
                nodes=tf.concat([g1.nodes,g2.nodes],axis = 1),
                edges=tf.concat([g1.edges,g2.edges],axis = 1),
                receivers=g1.receivers,
                senders=g1.senders,
                globals=tf.concat([g1.globals,g2.globals],axis =1)
        )
        return a

    def graph_add(self,g1,g2):

        a = gn.graphs.GraphsTuple(
            n_edge=g1.n_edge,
            n_node=g1.n_node,
            nodes=g1.nodes+g2.nodes,
            edges=g1.edges+g2.edges,
            receivers=g1.receivers,
            senders=g1.senders,
            globals=g1.globals+g2.globals
        )
        return a



    # def backtoimg(node_max,nodes):
#     with tf.tf.variable_scope('backtoimg' % (n + 1)):
#         for n in range(node_max):
#             with tf.variable_scope('block%d' % (n + 1)):
#                 nodes = tf.where()
#     return x













        ################################################################################################################
        ################################################################################################################







"""
Helper methods
"""

def conv(x, ksize, stride, filters_out, trainable=True):
    output = tf.layers.conv2d(
        inputs=x, filters=filters_out, kernel_size=ksize, strides=(stride, stride), padding='same',
        data_format='channels_last', activation=None, use_bias=False,
        kernel_initializer=tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
        bias_initializer=None, trainable=trainable)
    return output



def conv1(x, ksize, stride, filters_out, bn_train=True, use_bn=False):
    output = tf.layers.conv2d(
        inputs=x, filters=filters_out, kernel_size=ksize, strides=(stride, stride), padding='same',
        data_format='channels_last', activation=None, use_bias=False,
        kernel_initializer=tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(CONV_WEIGHT_DECAY),
        bias_initializer=None, trainable=True)
    if use_bn:
        output = bn1(output, is_training=bn_train)

    else:
        pass
    output = tf.nn.relu(output)
    return output

def stack(x, is_training, num_blocks, stack_stride, block_filters_internal, trainable=True):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, is_training, block_filters_internal=block_filters_internal, block_stride=block_stride, trainable=trainable)
    return x


def block(x, is_training, block_filters_internal, block_stride, trainable = True):
    filters_in = x.get_shape()[-1]

    m = 4
    filters_out = m * block_filters_internal
    shortcut = x

    with tf.variable_scope('a'):
        a_conv = conv(x, ksize=1, stride=block_stride, filters_out=block_filters_internal, trainable=trainable)
        a_bn = bn(a_conv, is_training)
        a = tf.nn.relu(a_bn)

    with tf.variable_scope('b'):
        b_conv = conv(a, ksize=3, stride=1, filters_out=block_filters_internal, trainable=trainable)
        b_bn = bn(b_conv, is_training)
        b = tf.nn.relu(b_bn)

    with tf.variable_scope('c'):
        c_conv = conv(b, ksize=1, stride=1, filters_out=filters_out, trainable=trainable)
        c = bn(c_conv, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut_conv = conv(x, ksize=1, stride=block_stride, filters_out=filters_out, trainable=trainable)
            shortcut = bn(shortcut_conv, is_training)

    return tf.nn.relu(c + shortcut)


def contains(target_str, search_arr):
    rv = False

    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break

    return rv


def bn(x, is_training, eps=1e-05, decay=BN_DECAY, affine=True):
    x = tf.layers.batch_normalization(inputs=x,training=is_training,axis=3,name='bn')#,momentum=BN_DECAY) #using default para
    A = tf.global_variables()
    return x

def bn1(x, is_training, eps=1e-05, decay=BN_DECAY, affine=True):
    x = tf.layers.batch_normalization(inputs=x,training=is_training,axis=3)#,momentum=BN_DECAY) #using default para

    return x











