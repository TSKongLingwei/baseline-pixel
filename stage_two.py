import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
from skimage import segmentation
from skimage import io

import graph_np  as gp
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

def create_label(label,segments):
    n_node = tf.reduce_max(segments)+1
    node_label = tf.cast(tf.unsorted_segment_mean(label, segments, n_node),
                         dtype=tf.float32)
    label_mean = tf.reduce_mean(tf.cast(label, dtype=tf.float32))

    a = tf.ones_like(node_label, dtype=tf.int32)
    b = tf.zeros_like(node_label, dtype=tf.int32)
    condition = tf.less(label_mean, node_label)
    label_t = tf.where(condition, a, b)
    label_t = tf.squeeze(label_t, axis=1)
    return label_t

def create_le(en,re,se,label_i):
    le = []
    for i in range(en[0]):
        if label_i[re[i]] == label_i[se[i]]:
            if label_i[re[i]]==0:
                le.append(1)
            if label_i[re[i]]==1:
                le.append(2)
        else:
            le.append(0)
    return np.array(le)


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

y = tf.placeholder(tf.int32, [batch_size, im_m, im_n, 1])
y1 = tf.placeholder(tf.int32,[None])
edge_label= tf.placeholder(tf.int32,[None])

z = tf.placeholder(tf.int32, [batch_size, im_m, im_n])



edge_num= tf.placeholder(tf.int32)
node_num = tf.placeholder(tf.int32)
receive= tf.placeholder(tf.int32,[None])
senders = tf.placeholder(tf.int32,[None])
edge_feature = tf.placeholder(tf.float32,[None,1])
global_fe = tf.placeholder(tf.float32,[1,32])
gf= np.zeros([1,32])
lr_ = tf.placeholder(tf.float32)

label_in = create_label(y,z)


model = ResNetModel(x=x,is_training=is_training, depth=50)

loss,loss1_mean ,loss2_mean,score1,score2=model.stage2_loss_test(x,y1,edge_label,z,edge_num,receive,senders,edge_feature,node_num,global_fe)
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
    model.load_original_weights(sess, skip_layers=['fc'])
    ###################################################################################
    #tf.train.Saver(var_list=tf.global_variables()).restore(sess, 'res/17')
    logfile = open("logfile2.txt", 'a')
    print("load Resnet.npy",file=logfile)
    logfile.close()
    print("load Resnet.npy")
    ###################################################################################
    iter_num = 10553
    lr = 1e-5
    counter = 0
    old_sumloss = 0
    for epoc in range(30):
        if epoc == 20:
            lr = lr / 10
        step = 0
        sum_loss = 0
        time_start = time.time()

        for num in range(iter_num):
            #########################################
            img, label, segments,AA = sess.run([X_train_batch_op, y_train_batch_op, segments_train_batch_op,A])



            label = np.int32(label)
            segments = np.int32(np.squeeze(segments,axis=3))
            nm[0] = np.max(segments)+1
            int_img=np.squeeze(img.astype(int))
            segments3 = np.squeeze(segments)
            en,re,se,ef = gp.np_edge_create(int_img,segments3)
            label_i = sess.run(label_in,{y: label, z: segments})
            le = create_le(en,re,se,label_i)



            nodefeture,edgefeature,loss_,loss1_m,loss2_m,_= sess.run([score1,score2,loss,loss1_mean,loss2_mean,update_op],
                                                    {x: img, y1: label_i, edge_label:le,z: segments,edge_num:en,receive:re,senders:se,edge_feature:ef,node_num:nm, lr_: lr,global_fe: gf,
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

                print("loss:", (sum_loss-old_sumloss)/300, "loss1:", loss1_m,"loss2:", loss2_m)
                old_sumloss = sum_loss
            #plt.imsave('1.png',np.uint8(np.squeeze(img)))
            #plt.imsave('2.png', np.uint8(np.squeeze(label))*255)
            #########################################



            step = step + 1
            ################################################
        gc.collect()
        time_end = time.time()
        ###################################################################################
        logfile = open("logfile2.txt", 'a')
        print('totally cost: ', (time_end - time_start), file=logfile)
        logfile.close()
        print('totally cost: ', (time_end - time_start))
        ###################################################################################
        ###################################################################################
        logfile = open("logfile2.txt", 'a')
        print('sum_loss:', (sum_loss / (iter_num)), 'epoch:',epoc,file=logfile)
        logfile.close()
        print('sum_loss:', (sum_loss / (iter_num)), 'epoch:',epoc)
        ###################################################################################
        if epoc % 1 == 0:
            tf.train.Saver(var_list=tf.global_variables()).save(sess, "row/" + str(epoc))
