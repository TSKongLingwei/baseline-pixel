import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
from up_sample_layer import up_sample_layer
import gc
import PIL.Image as Image
import time
import cv2
im_m=256
im_n=256
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
########################################################################################################################
########################################################################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
########################################################################################################################





########################################################################################################################
########################################################################################################################
########################################################################################################################
test_path = '/home/kong/Downloads/datasets/PASCALS/PASCALS-Image/'
test_image = os.listdir(test_path)
test_img_name = ['/home/kong/Downloads/datasets/PASCALS/PASCALS-Image/'+ x.strip() for x in test_image]
test_label_name = ['/home/kong/Downloads/datasets/PASCALS/PASCALS-Mask/'+ x.strip()[:-4] + '.png'for x in test_image]
out_path = ['/home/kong/Downloads/GCN/result/PASCALS/DGR/'+ x.strip()[:-4] + '.png'for x in test_image]
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
def data_augmentation(image,label,training=True):
    #image shape H,W,C
    #label shape H,W,C
    if training:
        # if np.round(np.random.rand(1)):
        if np.round(np.random.rand(1)):
            image[0, :, :, 0] = np.fliplr(image[0, :, :, 0])
            image[0, :, :, 1] = np.fliplr(image[0, :, :, 1])
            image[0, :, :, 2] = np.fliplr(image[0, :, :, 2])
            label[0, :, :, 0] = np.fliplr(label[0, :, :, 0])
            #label =np.array(np.squeeze(label),dtype=np.uint8)
            #plt.imshow(label)
            #plt.show()
            #image = image.transpose(Image.FLIP_LEFT_RIGHT) #左右对换。
            #label = label.transpose(Image.FLIP_LEFT_RIGHT)  # 左右对换。
        if np.round(np.random.rand(1)):
            image[0, :, :, 0] = np.flipud(image[0, :, :, 0])
            image[0, :, :, 1] = np.flipud(image[0, :, :, 1])
            image[0, :, :, 2] = np.flipud(image[0, :, :, 2])
            label[0, :, :, 0] = np.flipud(label[0, :, :, 0])
            #image = image.transpose(Image.FLIP_TOP_BOTTOM)  # 上下对换。
            #label = label.transpose(Image.FLIP_TOP_BOTTOM)  # 上下对换。
    return image, label
##################################################################################################################
##################################################################################################################
#mask12 = np.ones((im_M*im_N,im_M*im_N),dtype=np.float32) -  np.eye(im_M*im_N,dtype=np.float32)
# Model
# Placeholders
x = tf.placeholder(tf.float32, [1, im_m, im_n, 3])
y = tf.placeholder(tf.int32, [1, im_m, im_n])
y_48 = tf.placeholder(tf.int32, [1, 48, 48])
is_training = tf.placeholder('bool', [])
lr_ = tf.placeholder(tf.float32)
model = ResNetModel(x=x,is_training=is_training, depth=50)
loss, concat_up, cross_entropy_mean_net3, lap_loss = model.DGR(x, y, loss_weight=0.0)
update_op = model.optimize(learning_rate=lr_)
#A=tf.trainable_variables()
upscore_fuse = tf.nn.softmax(concat_up,3)
upscore_fuse = upscore_fuse[:,:,:,1]
#with tf.Session(config=config) as sess:
with tf.Session(config=config) as sess:
    ###################################################################################
    tf.train.Saver(var_list=tf.global_variables()).restore(sess, 'base/new_bn/20')
    logfile = open("testfile.txt", 'a')
    print("load Resnet.npy",file=logfile)
    logfile.close()
    print("load Resnet.npy")
    ###################################################################################
    FM = np.zeros(len(test_img_name))
    MAE = np.zeros(len(test_img_name))
    start_time = time.time()
    for step in range(len(test_img_name)):
        test_img = cv2.imread(test_img_name[step], cv2.IMREAD_COLOR)
        #img_rgb = np.zeros(test_img.shape,test_img.dtype)
        #img_rgb[:, :, 0] = test_img[:, :, 2]
        #img_rgb[:, :, 1] = test_img[:, :, 1]
        #img_rgb[:, :, 2] = test_img[:, :, 0]
        #test_img = img_rgb
        temp_0 = test_img.shape[0]
        temp_1 = test_img.shape[1]
        # cv2.imshow('img', test_img)
        # cv2.waitKey(0)
        test_img = cv2.resize(test_img, (im_m, im_n), interpolation=cv2.INTER_NEAREST)
        test_img = np.float32(test_img)
        test_label = cv2.imread(test_label_name[step], cv2.IMREAD_GRAYSCALE)
        test_img = test_img[np.newaxis, :]
        ####################################################################################################################
        map = sess.run(concat_up, {x: test_img, is_training: False})
        map = map[0,:,:,1]
        # map = np.squeeze(map)*1.3
        # map[map>1]=1
        # map[map<0]=0
        map = cv2.resize(np.squeeze(map), (temp_1, temp_0), interpolation=cv2.INTER_NEAREST)
        #map = (map - (1.0 - map)) * 1.1
        #map = (map - np.min(map)) / (np.max(map) - np.min(map))
        # cv2.imshow('img', map*255)
        # cv2.waitKey(0)
        #root = ["ECSSD/" + str(test_image[step][:-4]) + ".png"]
        #cv2.imwrite(root[0],map*255)
        #map = np.float32(cv2.imread(root[0], cv2.IMREAD_GRAYSCALE))
        map = np.rint(map * 255)
        cv2.imwrite(out_path[step], map)
        fm, mae = calculate_FM_MAE(test_label, map)
        ####################################################################################################################
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
