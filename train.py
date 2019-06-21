
import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
from up_sample_layer import up_sample_layer
import gc
import PIL.Image as Image
import matplotlib.pyplot as plt
import time
import cv2
im_m=256
im_n=256
c_image = 3
c_label = 1
batch_size = 16
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
########################################################################################################################
########################################################################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
########################################################################################################################
########################################################################################################################
def calculate_FM_MAE(truce,map):
    #########################################################################################
    map = np.float32(map)
    truce = np.float32(truce)
    map = np.squeeze(map)
    truce = np.squeeze(truce)
    ################################################
    temp = map.shape[0] * map.shape[1]
    MAE = np.sum(np.fabs(map - truce)/255) / temp
    ################################################
    truce = truce / 255
    map = map / 255
    truce[truce >= 0.1] = 1.0
    map = map / np.max(map)
    ################################################
    th1 = 2 * np.mean(map)
    if th1 > 1:
        th1 = 1
    map[map >= th1] = 1
    map[map < th1] = 0
    same_1_sum = np.sum(map * truce)
    pa_1 = same_1_sum / (np.sum(map) + 1e-8)
    ra_1 = same_1_sum / np.sum(truce)
    PreF = np.mean(pa_1)
    RecallF = np.mean(ra_1)
    FMeasureF = 1.3 * PreF * RecallF / (0.3 * PreF + RecallF + 1e-7)
    #########################################################################################
    return FMeasureF, MAE
###############################
def data_augmentation(image, label, training=True):
    if training:
        image_label = tf.concat([image, label], axis=-1)
        print('image label shape concat', image_label.get_shape())
        maybe_flipped = image_label
        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
        ################################################################################################################
        A = tf.random_uniform(shape=[1], minval=0, maxval=1)
        maybe_flipped = tf.cond(tf.greater(A[0], 0.5), lambda: tf.random_crop(
            tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(maybe_flipped, 0), size=[288, 288]), axis=0),
            size=[im_m, im_n, image_label.get_shape()[-1]]), lambda: maybe_flipped)
        ################################################################################################################
        image = maybe_flipped[:, :, :-1]
        mask = maybe_flipped[:, :, -1:]

        image = tf.image.random_brightness(image, 0.7)
        image = tf.image.random_hue(image, 0.3)
        # 设置随机的对比度
        image = tf.image.random_contrast(image,lower=0.3,upper=1.0)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)


        return image, mask, tf.greater(A[0], 0.5)


def read_csv(queue, augmentation=True):
    # csv = tf.train.string_input_producer(['./data/train/csv','./data/test.csv'])
    csv_reader = tf.TextLineReader(skip_header_lines=1)

    _, csv_content = csv_reader.read(queue)

    image_path, label_path = tf.decode_csv(csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    label_file = tf.read_file(label_path)

    image = tf.image.decode_image(image_file, channels=3)
    image.set_shape([im_m, im_n, c_image])
    image = tf.cast(image, tf.float32)

    label = tf.image.decode_image(label_file, channels=1)
    label.set_shape([im_m, im_n, c_label])

    label = tf.cast(label, tf.float32)
    # label = label / (tf.reduce_max(label) + 1e-7)
    #label = label / 255
    A = tf.random_uniform(shape=[1], minval=0, maxval=1)
    # 数据增强
    if augmentation:
        image, label, A = data_augmentation(image, label)
    else:
        pass
    return image, label, A
##################################################################################################################
train_csv = tf.train.string_input_producer(['DUTS-na.csv'])
train_image, train_label, A = read_csv(train_csv, augmentation=False)
X_train_batch_op, y_train_batch_op = tf.train.shuffle_batch([train_image, train_label], batch_size=batch_size,
                                                                capacity=batch_size * 2,
                                                                min_after_dequeue=batch_size * 1,
                                                                allow_smaller_final_batch=True)
##################################################################################################################
# Model
# Placeholders
is_training = tf.placeholder('bool', [])
x = tf.placeholder(tf.float32, [batch_size, im_m, im_n, 3])
y = tf.placeholder(tf.int32, [batch_size, im_m, im_n])
lr_ = tf.placeholder(tf.float32)
model = ResNetModel(x=x,is_training=is_training, depth=50)
loss, concat_up, lossf_mean, lap_loss = model.DGR(x, y, loss_weight=0.0)
update_op = model.optimize(learning_rate=lr_)

upscore_fuse = concat_up[:,:,:,1]
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #writer = tf.summary.FileWriter('log/logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    # Load the pretrained weights
    #model.load_original_weights(sess, skip_layers=['fc'])
    ###################################################################################
    tf.train.Saver(var_list=tf.global_variables()).restore(sess, 'base/new_bn/4')
    logfile = open("logfile.txt", 'a')
    print("load Resnet.npy",file=logfile)
    logfile.close()
    print("load Resnet.npy")
    ###################################################################################
    iter_num = 1000
    lr = 1e-5
    counter = 0
    for epoc in range(35):
        if epoc == 20:
            lr = lr / 10
        if epoc ==30:
            lr = lr/10
        step = 0
        sum_loss = 0
        sum_loss_old = 0
        time_start = time.time()
        for num in range(iter_num):
            #########################################
            img, label, AA = sess.run([X_train_batch_op, y_train_batch_op, A])
            label = np.int32(np.squeeze(label,axis=3))
            #plt.imsave('1.png',np.uint8(np.squeeze(img)))
            #plt.imsave('2.png', np.uint8(np.squeeze(label))*255)
            #########################################
            loss_v1, map, _, lap_loss_v1 = sess.run([lossf_mean, upscore_fuse, update_op, lap_loss],
                                       {x: img, y: label, lr_: lr, is_training: False})
            # counter += 1
            # writer.add_summary(summary,counter)
            # root = ["saliency_map/" + str(train_img[idxs[num]][:-4])+ '48' + ".png"]
            # cv2.imwrite(root[0],label_48*255)
            sum_loss = sum_loss + loss_v1*im_m*im_n
            if step % 50 == 0 or step == iter_num - 1:
                fm, mae = calculate_FM_MAE(label*255, map*255)
                ###################################################################################
                logfile = open("logfile.txt", 'a')
                print('epoc: {} step: {} loss_v1 = {} FM:{} MAE:{} lap_loss_v1:{}'.format(epoc, step + 1, loss_v1*im_m*im_n, fm, mae, lap_loss_v1), file=logfile)
                logfile.close()
                print('epoc: {} step: {} loss_v1 = {} FM:{} MAE:{} lap_loss_v1:{}'.format(epoc, step + 1, loss_v1*im_m*im_n, fm, mae, lap_loss_v1))
                ###################################################################################
            if step % 1000 == 0 or step == iter_num - 1:
                fm, mae = calculate_FM_MAE(label*255, map*255)
                logfile = open("logfile.txt", 'a')
                print('epoc: {} step: {} loss_1000 = {} '.format(epoc, step + 1,(sum_loss - sum_loss_old)/1000),file=logfile)
                logfile.close()
                print('epoc: {} step: {} loss_1000 = {} '.format(epoc, step + 1,(sum_loss - sum_loss_old)/1000))
                sum_loss_old = sum_loss

            step = step + 1

            ################################################
        gc.collect()
        time_end = time.time()
        ###################################################################################
        logfile = open("logfile.txt", 'a')
        print('totally cost: ', (time_end - time_start), file=logfile)
        logfile.close()
        print('totally cost: ', (time_end - time_start))
        ###################################################################################
        ###################################################################################
        logfile = open("logfile.txt", 'a')
        print('sum_loss:', (sum_loss / (iter_num)), file=logfile)
        logfile.close()
        print('sum_loss:', (sum_loss / (iter_num)))


        ###################################################################################
        if epoc % 2 == 0:
            tf.train.Saver(var_list=tf.global_variables()).save(sess, "base/new_bn/" + str(epoc))