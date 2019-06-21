import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
from skimage import segmentation
from skimage import io
import scipy.misc as misc
from graph_np import np_edge_create
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
batch_size = 1
segment_num = 256



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
########################################################################################################################
########################################################################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
############################################





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

    return image, label, A



def read_segcsv(queue, augmentation=True):
    # csv = tf.train.string_input_producer(['./data/train/csv','./data/test.csv'])
    csv_reader = tf.TextLineReader(skip_header_lines=1)

    _, csv_content = csv_reader.read(queue)

    image_path, label_path,segment_path = tf.decode_csv(csv_content, record_defaults=[[""], [""],[""]])

    image_file = tf.read_file(image_path)
    label_file = tf.read_file(label_path)
    segment_file = tf.read_file(segment_path)


    image = tf.image.decode_image(image_file, channels=3)
    image.set_shape([im_m, im_n, c_image])
    image = tf.cast(image, tf.float32)

    label = tf.image.decode_image(label_file, channels=1)
    label.set_shape([im_m, im_n, c_label])

    segments = tf.image.decode_image(segment_file, channels=1)
    segments.set_shape([im_m, im_n, c_label])

    label = tf.cast(label, tf.float32)
    segments = tf.cast(segments, tf.float32)
    # label = label / (tf.reduce_max(label) + 1e-7)
    #label = label / 255
    A = tf.random_uniform(shape=[1], minval=0, maxval=1)
    # 数据增强

    return image, label,segments, A




##################################################################################################################
train_csv = tf.train.string_input_producer(['DUTS.csv'])
train_image, train_label,train_segments, A = read_segcsv(train_csv, augmentation=False)
X_train_batch_op, y_train_batch_op,segments_train_batch_op = tf.train.shuffle_batch([train_image, train_label,train_segments], batch_size=batch_size,
                                                                capacity=batch_size * 3,
                                                                min_after_dequeue=batch_size * 1,
                                                                allow_smaller_final_batch=True)
##################################################################################################################
# Model
# Placeholders
is_training = tf.placeholder('bool', [])
x = tf.placeholder(tf.float32, [batch_size, im_m, im_n, 3])
base = tf.constant(np.full([1, 256, 256, 270], range(270)))
y = tf.placeholder(tf.int32, [batch_size, im_m, im_n,1])
#y = tf.placeholder(tf.int32, [batch_size, im_m, im_n])
z = tf.placeholder(tf.int32, [batch_size, im_m, im_n])
z2 = tf.placeholder(tf.int32, [batch_size, 128, 128])
edge_num= tf.placeholder(tf.int32)
node_num = tf.placeholder(tf.int32)
receive= tf.placeholder(tf.int32,[None])
senders = tf.placeholder(tf.int32,[None])
edge_feature = tf.placeholder(tf.float32,[None,1])
global_fe = tf.placeholder(tf.float32,[1,32])
lr_ = tf.placeholder(tf.float32)
model = ResNetModel(x=x,is_training=is_training, depth=50)
loss,cross_entropy_mean, node ,superlabel,attention=model.stage2_fuse_test(x,y,z,edge_num,receive,senders,edge_feature,node_num,base)
# loss, l1_map ,superlabel,loss1,loss2,loss3,loss4,loss5=model.stage2_fuse(x,y,z,z2,edge_num,receive,senders,edge_feature,node_num,base)
#g6,gl=model.stage2_loss(x,y,z,edge_num,receive,senders,edge_feature,node_num)
# f1,f2 = model.stage2_loss(x,y,z,edge_num,receive,senders,edge_feature,node_num)

update_op = model.optimize(learning_rate=lr_)


nm = np.array([1])








###########################################################


#################################################3
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #writer = tf.summary.FileWriter('log/logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    # Load the pretrained weights
    #model.load_original_weights(sess, skip_layers=['fc'])
    ###################################################################################
    tf.train.Saver(var_list=tf.global_variables()).restore(sess, 'sigmoid/3')
    logfile = open("logfile.txt", 'a')
    print("load Resnet.npy",file=logfile)
    logfile.close()
    print("load Resnet.npy")
    ###################################################################################
    iter_num = 10553
    lr = 1e-5
    counter = 0
    old_sumloss = 0
    time2 = 0
    time1 = 0
    for epoc in range(50):
        if epoc == 20:
            lr = lr / 10
        if epoc == 30:
            lr = lr / 10
        step = 0
        sum_loss = 0
        time_start = time.time()

        for num in range(iter_num):
            #########################################
            img, label, segments,AA = sess.run([X_train_batch_op, y_train_batch_op, segments_train_batch_op,A])

            label = np.int32(label)
            # label = np.int32(np.squeeze(label,axis=3))
            segments = np.int32(np.squeeze(segments,axis=3))
            nm[0] = np.max(segments)+1
            int_img=np.squeeze(img.astype(int))
            segments3 = np.squeeze(segments)
            en,re,se,ef = np_edge_create(int_img,segments3)
            # segments2=cv2.resize(segments3, (128, 128), interpolation=cv2.INTER_NEAREST)
            # segments2 =segments2[np.newaxis,:,:]


            # l1,loss_1,loss_2,loss_3,loss_4,loss_5,loss_,super_,_= sess.run([l1_map, loss1,loss2,loss3,loss4,loss5,loss,superlabel,update_op],
            #                                         {x: img, y: label, z: segments,z2:segments2,edge_num:en,receive:re,senders:se,edge_feature:ef,node_num:nm, lr_: lr,
            #                                         is_training: False})

            nodefeture,loss_,cross_entropy_mean_,super_,_,atten= sess.run([node, loss,cross_entropy_mean,superlabel,update_op,attention],
                                                                    {x: img, y: label, z: segments,
                                                                     edge_num: en, receive: re, senders: se,
                                                                     edge_feature: ef, node_num: nm, lr_: lr,
                                                                    is_training: False})

            # a1,a2= sess.run(
            #     [f1,f2],
            #     {x: img, y: label, z: segments, edge_num: en, receive: re, senders: se, edge_feature: ef, node_num: nm,
            #      lr_: lr,
            #      is_training: False})
            #aa,bb = sess.run([g6,gl], {x: img, y: label, z: segments,edge_num:en,receive:re,senders:se,edge_feature:ef,node_num:nm, lr_: lr,
            #                                        is_training: False})
            sum_loss = sum_loss+loss_
            if step % 300 == 0 or step == iter_num - 1:
                time2= time1
                time1 =time.time()
                print(time1-time2)
                print("loss:", (sum_loss-old_sumloss)*65536/300,"loss1:",cross_entropy_mean_*65536)

                old_sumloss = sum_loss
            #plt.imsave('1.png',np.uint8(np.squeeze(img)))
            #plt.imsave('2.png', np.uint8(np.squeeze(label))*255)
            #########################################



            step = step + 1
            ################################################
        gc.collect()
        time_end = time.time()
        ###################################################################################
        logfile = open("logfile3.txt", 'a')
        print('totally cost: ', (time_end - time_start),"epoch:",epoc ,"lr:",lr,file=logfile)
        logfile.close()
        print('totally cost: ', (time_end - time_start),"epoch:",epoc ,"lr:",lr)
        ###################################################################################
        ###################################################################################
        logfile = open("logfile3.txt", 'a')
        print('sum_loss:', (sum_loss*65536 / (iter_num)), file=logfile)
        logfile.close()
        print('sum_loss:', (sum_loss*65536 / (iter_num)))
        ###################################################################################
        if epoc % 1 == 0:
            tf.train.Saver(var_list=tf.global_variables()).save(sess, "sigmoid/" + str(epoc))
