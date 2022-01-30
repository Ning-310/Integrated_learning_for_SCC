# -*- coding: utf-8 -*-
# @Time    : 2021/2/10  14:09
# @Author  : Gou Yujie
# @File    : area_classify.py

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,MaxPooling2D
from keras.utils import to_categorical
import cv2
import numpy as np
import os
from keras.callbacks import ModelCheckpoint,EarlyStopping
import re
from tensorflow.keras.metrics import AUC
from PIL import Image
from sklearn.model_selection import StratifiedKFold
def model(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(0.5)(pool5)
    pool5 = BatchNormalization()(pool5)

    conv6 = Dropout(0.5)(pool5)

    f1=Flatten()(conv6)
    dense1 = Dense(4096, activation='relu')(f1)
    bn1 = BatchNormalization()(dense1)
    ac1 = Activation('relu')(bn1)
    drop1 = Dropout(0.4)(ac1)
    dense3=Dense(2,activation='softmax')(drop1)

    model = Model(inputs=[inputs], outputs=[dense3])
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=[AUC(name='auc')])
    print(model.summary())
    return model

def load_data(label_folder):
    images = []
    label_red_cancer = []
    label_blue_keratin=[]
    label_green_bloodvessel = []
    label_bluegreen_mid = []
    for root, dirs, files in os.walk(label_folder):
        for file in files:
            if file.endswith(".txt"):
                continue
            imagepath = os.path.join(root, file)
            if re.findall('original',imagepath):
                print(imagepath)
                ori = Image.open(imagepath)
                images.append(np.array(ori, np.uint8))
            elif re.findall('labels',imagepath):
                img = Image.open(imagepath)
                img=np.array(img,np.uint8)
                r = np.array([255, 0, 0])
                blu = np.array([0, 0, 255])
                gre = np.array([0, 255, 0])
                bg = np.array([0, 255, 255])
                mask1 = cv2.inRange(img, r, r)
                mask2 = cv2.inRange(img, blu, blu)
                mask3 = cv2.inRange(img, gre, gre)
                mask4 = cv2.inRange(img, bg, bg)
                if np.sum(mask1)>0:
                    label_red_cancer.append(1)
                else:
                    label_red_cancer.append(0)
                if np.sum(mask4)>0:
                    label_bluegreen_mid.append(1)
                else:
                    label_bluegreen_mid.append(0)
                if np.sum(mask3)>0:
                    label_green_bloodvessel.append(1)
                else:
                    label_green_bloodvessel.append(0)
                if np.sum(mask2)>0:
                    label_blue_keratin.append(1)
                else:
                    label_blue_keratin.append(0)
            else:
                pass
    images = np.array(images)
    label_red_cancer = np.array(to_categorical(label_red_cancer,2))
    label_bluegreen_mid = np.array(to_categorical(label_bluegreen_mid,2))
    label_green_bloodvessel = np.array(to_categorical(label_green_bloodvessel,2))
    label_blue_keratin = np.array(to_categorical(label_blue_keratin,2))

    return images, label_red_cancer,label_bluegreen_mid,label_green_bloodvessel,label_blue_keratin

if __name__=='__main__':
    w = h = 256
    c = 3
    lab_folder="F:\cervical_cancer/area_classify/all/"
    images, label_red_cancer,label_bluegreen_mid,label_green_bloodvessel,label_blue_keratin=load_data(lab_folder)
    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    count=0
    cla = []
    for train, test in sfolder.split(images, label_red_cancer):
        count += 1
        x_train, x_test = images[train], images[test]
        y_train, y_test = label_red_cancer[train], label_red_cancer[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        model=model((w,h,c))

        print('----------------cancer training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint(f'models/classify_red_cancer_%d.model'%(count), save_weights_only=False, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=5, epochs=15, validation_data=(x_test, y_test),callbacks=[checkpoint,earlystopping], verbose=1,class_weight={0:3414,1:4429})

    count = 0
    cla = []
    for train, test in sfolder.split(images, label_blue_keratin):
        count += 1
        x_train, x_test = images[train], images[test]
        y_train, y_test = label_blue_keratin[train], label_blue_keratin[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        model = model((w, h, c))
        print('----------------keratin training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint(f'models/classify_blue_keratin_%d.model'%(count), save_weights_only=False, monitor='val_loss',verbose=1,
                                     save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=5, epochs=15, validation_data=(x_test, y_test),callbacks=[checkpoint,earlystopping],verbose=1, class_weight={0: 819, 1: 7024})

    count = 0
    cla = []
    for train, test in sfolder.split(images, label_bluegreen_mid):
        count += 1
        x_train, x_test = images[train], images[test]
        y_train, y_test = label_bluegreen_mid[train], label_bluegreen_mid[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        model = model((w, h, c))
        print('----------------stroma training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint(f'models/classify_bluegreen_mid_%d.model'%(count), save_weights_only=False, monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=5, epochs=15, validation_split=0.2, callbacks=[checkpoint,earlystopping], verbose=1,
                  class_weight={0: 2244, 1: 5599})
    count = 0
    cla = []
    for train, test in sfolder.split(images, label_green_bloodvessel):
        count += 1
        x_train, x_test = images[train], images[test]
        y_train, y_test = label_green_bloodvessel[train], label_green_bloodvessel[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        model = model((w, h, c))
        print('----------------bloodvessel training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint(f'models/classify_green_bloodvessel_%d.model'%(count), save_weights_only=False, monitor='val_loss',verbose=1,
                                     save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=5, epochs=15, validation_split=0.2, callbacks=[checkpoint,earlystopping],verbose=1, class_weight={0: 1366, 1: 6477})