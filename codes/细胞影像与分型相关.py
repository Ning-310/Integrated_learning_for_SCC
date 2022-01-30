# -*- coding: utf-8 -*-
# @Time    : 2021/7/20  11:53
# @Author  : Gou Yujie
# @File    : 细胞影像与分型相关.py

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import cv2
import numpy as np
import os
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
from PIL import Image,ImageFile
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
def model(input_size,l):
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

    conv5 = Dropout(0.5)(pool3)
    f1=Flatten()(conv5)
    dense1 = Dense(4096, activation='relu')(f1)
    bn1 = BatchNormalization()(dense1)
    ac1 = Activation('relu')(bn1)
    drop1 = Dropout(0.4)(ac1)
    dense3=Dense(l,activation='softmax')(drop1)

    model = Model(inputs=[inputs], outputs=[dense3])
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=[AUC(name='auc')])
    print(model.summary())
    return model
def conf_auc(ground_truth,test_predictions,l):
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(int(l)):
        fpr[i], tpr[i], _ = metrics.roc_curve(ground_truth[:, i], test_predictions[:, i],pos_label=1)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    return fpr,tpr,roc_auc

file=pd.read_csv('subtype.txt',sep='\t',header=0)
id_subtype=dict(zip(file['sample_id'],file['subt']))
file2=pd.read_csv('clinical_data.txt',sep='\t',header=0)
id_grade2=dict(zip(file2['sample_id'],file2['Stage_FIGO2018']))
trans_dic={'IIIB':0,'IIB':1,'IIIC1r':2,'IIA2':3,'IIA1':4,'IB1':5,'IB2':6,'IB3':7,'IIIC1p':8}
for k,v in id_grade2.items():
    id_grade2[k]=trans_dic[v]
outpic=open("hpv_dic_allpics.txt",'rb')
allpics_mooc=pickle.load(outpic)
sub_label=[]
sub_pics=[]
grade_label=[]
grade_pics=[]

for k in allpics_mooc.keys():
    mooc = k.split('~')[0]
    if mooc in id_subtype.keys():
        print(mooc)
        sub_pics+=allpics_mooc[k]
        sub_label+=([id_subtype[mooc]] * len(allpics_mooc[k]))
for k in allpics_mooc.keys():
    mooc = k.split('~')[0]
    if mooc in id_grade2.keys():
        print(mooc)
        grade_pics+=allpics_mooc[k]
        grade_label+=([id_grade2[mooc]] * len(allpics_mooc[k]))
igo = [sub_label,grade_label]
lens = [3,9]
names = ['subtype','grade_2018']
pics_lis = [sub_pics,grade_pics]
sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
all_pred_bloodvessel = []
all_true_bloodvessel = []
count = 0
kind = 0
count_auc_green = {}
w = h = 128
c = 3
for num, i in enumerate(igo):
    model = model((w, h, c), lens[num])
    name = names[num]
    for train, test in sfolder.split(pics_lis[num], i):
        count = count + 1
        images_green = np.array(pics_lis[num])
        label_green_bloodvessel = np.array(i)
        x_train, x_test = images_green[train], images_green[test]
        y_train, y_test = label_green_bloodvessel[train], label_green_bloodvessel[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        if os.path.exists("./models/%s/%d.model" % (name, count)):
            model_pre = load_model('./models/%s/%s.model' % (name, name))
            x_pred = list(model_pre.predict(x_test))
            all_pred_bloodvessel = all_pred_bloodvessel + list(x_pred)
            all_true_bloodvessel = all_true_bloodvessel + list(y_test)
            continue

        print('----------------cell_subtype training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint('./models/%s/%s.model' % (name, name), save_weights_only=False,
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=200, epochs=25, validation_data=(x_test, y_test),
                  callbacks=[checkpoint, earlystopping], verbose=1)
        model_pre = load_model('./models/%s/%s.model' % (name, name))
        x_pred = list(model_pre.predict(x_test))
        all_pred_bloodvessel = all_pred_bloodvessel + list(x_pred)
        all_true_bloodvessel = all_true_bloodvessel + list(y_test)
        fpr, tpr, aucdic = conf_auc(y_test, np.array(x_pred), lens[num])
        with open('./models/%s/dic_fpr_%s_%d.txt' % (name, name, count), 'wb') as fout:
            pickle.dump(fpr, fout)
        with open('./models/%s/dic_tpr_%s_%d.txt' % (name, name, count), 'wb') as tout:
            pickle.dump(tpr, tout)
        with open('./models/%s/dic_auc_%s_%d.txt' % (name, name, count), 'wb') as aout:
            pickle.dump(aucdic, aout)
        fout.close()
        tout.close()
        aout.close()
        model_pre.save("./models/%s/%d.model" % (name, count))
        print(aucdic)
    fpr, tpr, aucdic = conf_auc(np.array(all_true_bloodvessel), np.array(all_pred_bloodvessel), lens[num])
    with open('dic_fpr_%s.txt' % (name), 'wb') as fout:
        pickle.dump(fpr, fout)
    with open('dic_tpr_%s.txt' % (name), 'wb') as tout:
        pickle.dump(tpr, tout)
    with open('dic_auc_%s.txt' % (name), 'wb') as aout:
        pickle.dump(aucdic, aout)
    print(aucdic)