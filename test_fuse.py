import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
from skimage import segmentation
from skimage import io

from graph_np import np_edge_create
from graph_np import node2img
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


# test_path = '/home/kong/Downloads/caffe-sal/data/DUT-train/im/'
# test_image = os.listdir(test_path)
# test_img_name = ['/home/kong/Downloads/caffe-sal/data/DUT-train/im/'+ x.strip() for x in test_image]
# test_label_name = ['/home/kong/Downloads/caffe-sal/data/DUT-train/mask/'+ x.strip()[:-4] + '.png'for x in test_image]
# test_super_name = ['/home/kong/Downloads/caffe-sal/data/DUT-train/segments/'+ x.strip()[:-4] + '.png'for x in test_image]


test_path = '/home/kong/Downloads/datasets/PASCALS/PASCALS-Image/'
test_image = os.listdir(test_path)
test_img_name = ['/home/kong/Downloads/datasets/PASCALS/PASCALS-Image/'+ x.strip() for x in test_image]
test_label_name = ['/home/kong/Downloads/datasets/PASCALS/PASCALS-Mask/'+ x.strip()[:-4] + '.png'for x in test_image]
test_super_name = ['/home/kong/Downloads/datasets/PASCALS/256/'+ x.strip()[:-4] + '.png'for x in test_image]
out_path = ['/home/kong/Downloads/GCN/result/PASCALS/fuse-FCN/'+ x.strip()[:-4] + '.png'for x in test_image]

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
    a = np.max(map)
    if a ==0:
        a = 1
    map = map /a
    ################################################
    th1 = 2 * np.mean(map)
    if th1 > 1:
        th1 = 1
    if th1 ==0:
        th1 = 0.1
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

##################################################################################################################
# Model
# Placeholders
is_training = tf.placeholder('bool', [])
x = tf.placeholder(tf.float32, [batch_size, im_m, im_n, 3])
base = tf.constant(np.full([1, 256, 256, 270], range(270)))
y = tf.placeholder(tf.int32, [batch_size, im_m, im_n,1])
#y = tf.placeholder(tf.int32, [batch_size, im_m, im_n])
z = tf.placeholder(tf.int32, [batch_size, im_m, im_n])
# z2 = tf.placeholder(tf.int32, [batch_size, 128, 128])
edge_num= tf.placeholder(tf.int32)
node_num = tf.placeholder(tf.int32)
receive= tf.placeholder(tf.int32,[None])
senders = tf.placeholder(tf.int32,[None])
edge_feature = tf.placeholder(tf.float32,[None,1])
global_fe = tf.placeholder(tf.float32,[1,32])
lr_ = tf.placeholder(tf.float32)
model = ResNetModel(x=x,is_training=is_training, depth=50)

# loss, l1_map ,superlabel,loss1,loss2,loss3,loss4,loss5=model.stage2_fuse_test(x,y,z,edge_num,receive,senders,edge_feature,node_num,base)
loss,cross_entropy_mean, node ,superlabel,attention=model.stage2_fuse_test(x,y,z,edge_num,receive,senders,edge_feature,node_num,base)
#g6,gl=model.stage2_loss(x,y,z,edge_num,receive,senders,edge_feature,node_num)
# f1,f2 = model.stage2_loss(x,y,z,edge_num,receive,senders,edge_feature,node_num)

update_op = model.optimize(learning_rate=lr_)


nm = np.array([1])






###########################################################


#################################################3
with tf.Session(config=config) as sess:
    ###################################################################################
    tf.train.Saver(var_list=tf.global_variables()).restore(sess, 'sigmoid/25')
    logfile = open("testfile.txt", 'a')
    print("load Resnet.npy",file=logfile)
    logfile.close()
    print("load Resnet.npy")
    ###################################################################################
    FM = np.zeros(len(test_img_name))
    MAE = np.zeros(len(test_img_name))
    start_time = time.time()

    ###################################################################################
    for step in range(len(test_img_name)):
        test_img = cv2.imread(test_img_name[step], cv2.IMREAD_COLOR)

        temp_0 = test_img.shape[0]
        temp_1 = test_img.shape[1]
        # cv2.imshow('img', test_img)
        # cv2.waitKey(0)
        test_img = cv2.resize(test_img, (im_m, im_n), interpolation=cv2.INTER_NEAREST)
        test_img = np.float32(test_img)
        test_img = test_img[np.newaxis, :]


        test_label = cv2.imread(test_label_name[step], cv2.IMREAD_GRAYSCALE)

        segments = cv2.imread(test_super_name[step], cv2.IMREAD_GRAYSCALE)

        input_label = cv2.resize(test_label, (im_m, im_n), interpolation=cv2.INTER_NEAREST)



            #########################################




        label = np.int32(input_label / 255)
        label = label[np.newaxis, :, :,np.newaxis]
        segments = np.int32(segments)
        segments = segments[np.newaxis, :, :]
        nm[0] = np.max(segments) + 1
        int_img = np.squeeze(test_img.astype(int))
        segments3 = np.squeeze(segments)
        en, re, se, ef = np_edge_create(int_img, segments3)
        # segments2 = cv2.resize(segments3, (128, 128), interpolation=cv2.INTER_NEAREST)
        # segments2 = segments2[np.newaxis, :, :]
        l1map,atten = sess.run(
            [node, attention],
            {x: test_img, z: segments,
             edge_num: en, receive: re, senders: se,
             edge_feature: ef, node_num: nm,is_training: False})

        # l1map,attention= sess.run(
        #     [node,attention],
        #     {x: test_img, y: label, z: segments,  edge_num: en, receive: re, senders: se, edge_feature: ef,
        #      node_num: nm,
        #      is_training: False})

        # atten1 = atten[0,:,:,1]
        fuse_map = np.squeeze(l1map,axis = 0)
        forground = fuse_map[:,:,0]
        back_ground = fuse_map[:,:,1]

        fuse = back_ground
        fuse[fuse>1] =1
        fuse[fuse<0] =0

        # f= np.mean(atten1)
        # atten1[atten1>=f]=1
        # atten1[atten1 <f] = 0

        fuse = cv2.resize(np.squeeze(fuse), (temp_1, temp_0), interpolation=cv2.INTER_NEAREST)
        # fuse = cv2.resize(np.squeeze(atten1), (temp_1, temp_0), interpolation=cv2.INTER_NEAREST)
        fuse = fuse *255
        cv2.imwrite(out_path[step],fuse)

            # a1,a2= sess.run(
            #     [f1,f2],
            #     {x: img, y: label, z: segments, edge_num: en, receive: re, senders: se, edge_feature: ef, node_num: nm,
            #      lr_: lr,
            #      is_training: False})
            #aa,bb = sess.run([g6,gl], {x: img, y: label, z: segments,edge_num:en,receive:re,senders:se,edge_feature:ef,node_num:nm, lr_: lr,
            #                                        is_training: False})

        fm, mae = calculate_FM_MAE(test_label, fuse)
        # ####################################################################################################################
        FM[step] = fm
        MAE[step] = mae
        if step % 500 == 0:
            ###################################################################################
            logfile = open("testfile.txt", 'a')
            print("step: {}".format(step), file=logfile)
            logfile.close()
            print("step: {}".format(step))
            ###################################################################################
            ####################################################################################################################
    gc.collect()
    #########################################################################################
    ###################################################################################
    logfile = open("testfile.txt", 'a')
    print("FMeasure = {}  MAE = {}".format(np.mean(FM), np.mean(MAE)), file=logfile)
    logfile.close()
    print("FMeasure = {}  MAE = {}".format(np.mean(FM), np.mean(MAE)))
    ###################################################################################
    diff_time = time.time() - start_time
    print('Detection took {}s per image'.format(diff_time / len(test_img_name)))
    ###################################################################################