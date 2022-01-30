# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 19:51
# @Author  : Gou Yujie
# @File    : model.py

from tensorflow.keras.models import Model
from keras.optimizers import Adam,RMSprop
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPool2D,BatchNormalization,Activation,Input,Conv2DTranspose,Concatenate,concatenate,UpSampling2D,AveragePooling2D,SeparableConv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import metrics
import cv2
from keras.callbacks import Callback,ModelCheckpoint
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from tensorflow.keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
def get_unet(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='sigmoid', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, (3, 3), activation='sigmoid', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
        2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(conv6), conv3], axis=3)
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(conv7), conv2], axis=3)
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(
        2, 2), padding='same')(conv8), conv1], axis=3)
    up9 = BatchNormalization()(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='sigmoid', padding='same')(conv9)

    conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(2, (1, 1), activation='sigmoid')(conv9)

    conv10 = Dropout(0.4)(conv10)
    conv10 = Activation('softmax')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=[AUC(name='auc')])
    print(model.summary())
    return model

def load_data(picfolder,labelfolder):
    image_files = []
    mask_files = []
    for root, dirs, files in os.walk(picfolder):
        for file in files:
            imgpath = os.path.join(root,file)
            img=cv2.imread(imgpath)
            (B, G, R) = cv2.split(img)
            img = cv2.merge([R, G, B])
            img = cv2.resize(img, (w, h))
            img = img.reshape((w, h, 3))
            image_files.append(img)
    for root, dirs, files in os.walk(labelfolder):
        for file in files:
            labelpath = os.path.join(root, file)
            label = cv2.imread(labelpath)  #label是三通道，灰度图是单通道
            mask=cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (w, h))
            mask[mask<100]=0
            mask[mask >= 100] = 1 #1表示白色，细胞部分
            mask_files.append(mask)

    return image_files,mask_files

def predict_and_process(data,model):
    for loc,image in enumerate(data):
        print(image.shape)
        image=np.uint8(image)
        pred = model.predict(image.reshape(1,w,h,c))  # float32 (1,256,256,2)
        pred = pred[:, :, 1].reshape((w, h))
        pred = pred * 255
        pred=np.uint8(pred)
        pred = np.uint8(pred)
        binary = cv2.adaptiveThreshold(pred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_erode = cv2.erode(binary, kernel, iterations=3)
        si = cv2.medianBlur(img_erode, 7)
        si2 = cv2.medianBlur(si, 3)
        img_dilated = cv2.dilate(si2, kernel, iterations=3)
        contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for k in range(len(contours)):
            M2 = cv2.moments(contours[k])
            if M2["m00"] != 0:
                center_x2 = int(M2["m10"] / M2["m00"])
                center_y2 = int(M2["m01"] / M2["m00"])
                if(contours[k].size>15):
                    cv2.circle(image, (center_x2, center_y2), 2, (0, 255, 0), -1)
        image.save("F:\cervical_cancer/test/addmark/%d.png"%(loc))

if __name__=='__main__':

    picfolder="F:\cervical_cancer/test\predict_to_pics/"
    labelfolder="F:\cervical_cancer/test\predict_to_label/"
    w=h=128
    c=3
    image_files, mask_files= load_data(picfolder,labelfolder)
    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    count = 0
    cla = []
    for train, test in sfolder.split(image_files, mask_files):
        count += 1
        x_train, x_test = image_files[train], image_files[test]
        y_train, y_test = mask_files[train], mask_files[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        model = get_unet((w,h,c))
        print('----------------training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint(f'models/get_unet_%d.h5'%(count), save_weights_only=False, monitor='val_loss',
                                     save_best_only=True, mode='auto',verbose=1)
        model.fit(x_train, y_train, batch_size=30, epochs=25,validation_data=(x_test, y_test),
                  callbacks=[checkpoint, earlystopping], verbose=1)
        predicts=predict_and_process(x_test,model)




