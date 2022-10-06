"""

Created on Sun Dec 29 19:21:08 2019

@author: xiuzhang Eastmount CSDN

"""
import csv

import os

import glob

import cv2

import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

# 定義圖片路徑

path = 'photo/'
# path_test = 'orchid_private_set/'
path_test = 'photo/'

#---------------------------------第一步 讀取圖像-----------------------------------

def read_img(path):

    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]

    imgs = []

    labels = []

    fpath = []

    for idx, folder in enumerate(cate):

        # 遍曆整個目錄判斷每個文件是不是符合

        for im in glob.glob(folder + '/*.jpg'):

            #print('reading the images:%s' % (im))

            img = cv2.imread(im)             #調用opencv庫讀取像素點

            img = cv2.resize(img, (32, 32))  #圖像像素大小一致

            imgs.append(img)                 #圖像數據

            labels.append(idx)               #圖像類標

            fpath.append(path+im)            #圖像路徑名

            #print(path+im, idx)

    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

def real_test_read_img(path):

    imgs = []

    labels = []

    fpath = []

    for im in glob.glob(path + '*.jpg'):

        #print('reading the images:%s' % (im))

        img = cv2.imread(im)             #調用opencv庫讀取像素點

        img = cv2.resize(img, (32, 32))  #圖像像素大小一致

        imgs.append(img)                 #圖像數據

        labels.append(0)                 #圖像類標

        fpath.append(path+im)            #圖像路徑名

        print(path+" "+im)

    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

# 讀取圖像

# fpaths, data, label = read_img(path)

# fpathsT, dataT, labelT = real_test_read_img(path_test)

# print(data.shape)  # (1000, 256, 256, 3)

# 計算有多少類圖片

num_classes = 219

print("num_classes:",num_classes)

# x_train = data

# y_train = label

# fpaths_train = fpaths

# x_val = dataT

# y_val = labelT

# fpaths_test = fpathsT

# print(len(x_train),len(y_train),len(x_val),len(y_val)) #800 800 200 200

# print(y_val)

#---------------------------------第二步 建立神經網絡-----------------------------------

# 定義Placeholder

xs = tf.placeholder(tf.float32, [None, 32, 32, 3])  #每張圖片32*32*3個點

ys = tf.placeholder(tf.int32, [None])               #每個樣本有1個輸出

# 存放DropOut參數的容器

drop = tf.placeholder(tf.float32)                   #訓練時為0.25 測試時為0

# 定義卷積層 conv0

conv0 = tf.layers.conv2d(xs, 20, 5, activation=tf.nn.relu)    #20個卷積核 卷積核大小為5 Relu激活

# 定義max-pooling層 pool0

pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])        #pooling窗口為2x2 步長為2x2

print("Layer0：\n", conv0, pool0)

# 定義卷積層 conv1

conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu) #40個卷積核 卷積核大小為4 Relu激活

# 定義max-pooling層 pool1

pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])        #pooling窗口為2x2 步長為2x2

print("Layer1：\n", conv1, pool1)

# 將3維特征轉換為1維向量

flatten = tf.layers.flatten(pool1)

# 全連接層 轉換為長度為400的特征向量

fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

print("Layer2：\n", fc)

# 加上DropOut防止過擬合

dropout_fc = tf.layers.dropout(fc, drop)

# 未激活的輸出層

logits = tf.layers.dense(dropout_fc, num_classes)

print("Output：\n", logits)

# 定義輸出結果

predicted_labels = tf.arg_max(logits, 1)

#---------------------------------第三步 定義損失函數和優化器---------------------------------

# 利用交叉熵定義損失

losses = tf.nn.softmax_cross_entropy_with_logits(

        labels = tf.one_hot(ys, num_classes),       #將input轉化為one-hot類型數據輸出

        logits = logits)

# 平均損失

mean_loss = tf.reduce_mean(losses)

# 定義優化器 學習效率設置為0.0001

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(losses)

#------------------------------------第四步 模型訓練和預測-----------------------------------

# 用於保存和載入模型

saver = tf.train.Saver()

# 訓練或預測

train = False

# 模型文件路徑

model_path = "model/image_model"

with tf.Session() as sess:

    print("測試模式")

    # 測試載入參數

    saver.restore(sess, model_path)

    print("從{}載入模型".format(model_path))

    # label和名稱的對照關系

    label_name_dict = {

    }
    for i in range(219):
        label_name_dict[i] = str(i)
    
    counter = 0
    with open('output.csv', 'w', newline='') as csvfile:
        counter+=1
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        # 寫入一列資料
        writer.writerow(['filename', 'category'])
        t = []
        p = []
        for index in range(219):
            for im in glob.glob(path_test+str(index)+"/" + '*.jpg'):

                #print('reading the images:%s' % (im))

                img = cv2.imread(im)             #調用opencv庫讀取像素點

                img = cv2.resize(img, (32, 32))  #圖像像素大小一致

                imgs = []

                imgs.append(img)                 #圖像數據

                # print(im)
            # 定義輸入和Label以填充容器 測試時dropout為0
                
                # print(np.asarray(imgs, np.float32).shape)

                test_feed_dict = {

                    xs: np.asarray(imgs, np.float32),

                    ys: [0],

                    drop: 0

                }

            # 真實label與模型預測label
            
                predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
                
                predicted_label = predicted_labels_val[0]

                # 將label id轉換為label名

                predicted_label_name = label_name_dict[predicted_label]

                # 寫入另外幾列資料
                writer.writerow([str(im)[-14:], predicted_label_name])
                print(str(im)[-14:]," ",index, " ", predicted_label_name)
                t.append(index)
                p.append(int(predicted_label_name))
            # print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
        print(sum(1 for x,y in zip(t,p) if x == y))
        print(len(t))
        print(sum(1 for x,y in zip(t,p) if x == y) / len(t))
    # 評價結果

    # print("正確預測個數:", sum(y_val==predicted_labels_val))

    # print("准確度為:", 1.0*sum(y_val==predicted_labels_val) / len(y_val))