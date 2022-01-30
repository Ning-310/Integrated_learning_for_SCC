# -*- coding: utf-8 -*-
# @Time    : 2021/7/18  17:18
# @Author  : Gou Yujie
# @File    : 细胞影像与分期关联2_linux.py
# -*- coding: utf-8 -*-
# @Time    : 2021/7/15  10:32
# @Author  : Gou Yujie
# @File    : 细胞影像与分期关联.py

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,MaxPooling2D
from keras.utils import to_categorical
from sklearn import metrics
from keras.callbacks import ModelCheckpoint,EarlyStopping
import cv2
import numpy as np
import os
from tensorflow.keras.metrics import AUC
from keras.models import load_model
from PIL import Image,ImageFile
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

def load_pic(picpath,w,w2):
    origin_pic = Image.open(picpath)
    w_pic = origin_pic.size[0]
    h_pic = origin_pic.size[1]
    origin_pic = np.array(origin_pic)
    pics = []
    for x in range(w_pic // w):
        for y in range(h_pic // w):
            if (x + 1) * w <= w_pic and (y + 1) * w <= h_pic:
                pic = origin_pic[y * w:(y + 1) * w, x * w:(x + 1) * w]
                _, gray_G, _ = cv2.split(pic)
                if (gray_G > 200).all():
                    continue
                pic=Image.fromarray(pic.astype('uint8')).convert('RGB')
                pic = pic.resize((w2, w2), Image.ANTIALIAS)
                pic = np.array(pic)/255
                pics.append(pic)
    return pics

def conf_auc(ground_truth,test_predictions,l):
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(int(l)):
        fpr[i], tpr[i], _ = metrics.roc_curve(ground_truth[:, i], test_predictions[:, i],pos_label=1)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    return fpr,tpr,roc_auc


name_mocc=open("name-mocc.txt",'r',encoding='gbk')
outpic=open("allpics.txt",'rb')
pics=pickle.load(outpic)
out_stages=open("allstages.txt",'rb')
stages=pickle.load(out_stages)
out_grades=open("allgrades.txt",'rb')
grades=pickle.load(out_grades)
out_lyms=open("alllyms.txt",'rb')
lyms=pickle.load(out_lyms)

igo=[stages,grades,lyms]
lens=[6,5,2]
names=['stages','grades','lyms']
sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
all_pred_bloodvessel=[]
all_true_bloodvessel=[]
count=0
kind=0
count_auc_green={}
w=h=128
c=3
for num,i in enumerate(igo):
    model = model((w, h, c),lens[num])
    name = names[num]
    for train, test in sfolder.split(pics,i):
        count = count + 1
        images_green = np.array(pics)
        label_green_bloodvessel = np.array(i)
        x_train, x_test = images_green[train], images_green[test]
        y_train, y_test = label_green_bloodvessel[train], label_green_bloodvessel[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        if os.path.exists("./models/%s/%d.model"%(name,count)):
            model_pre = load_model("./models/%s/%d.model" % (name, count))
            x_pred = list(model_pre.predict(x_test))
            all_pred_bloodvessel = all_pred_bloodvessel + list(x_pred)
            all_true_bloodvessel = all_true_bloodvessel + list(y_test)
            continue

        print('----------------cell_clinical training-----------------------')

        earlystopping=EarlyStopping(monitor='val_loss',patience=10,mode='auto')
        checkpoint = ModelCheckpoint('./models/%s/%s.model'%(name,name), save_weights_only=False, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=100, epochs=25, validation_data=(x_test, y_test), callbacks=[checkpoint,earlystopping], verbose=1)
        model_pre=load_model('./models/%s/%s.model'%(name,name))
        x_pred=list(model_pre.predict(x_test))
        all_pred_bloodvessel=all_pred_bloodvessel+list(x_pred)
        all_true_bloodvessel=all_true_bloodvessel+list(y_test)
        fpr, tpr, aucdic = conf_auc(y_test, np.array(x_pred),lens[num])
        with open('./models/%s/dic_fpr_%s_%d.txt' % (name,name,count), 'wb') as fout:
            pickle.dump(fpr, fout)
        with open('./models/%s/dic_tpr_%s_%d.txt' % (name,name,count), 'wb') as tout:
            pickle.dump(tpr, tout)
        with open('./models/%s/dic_auc_%s_%d.txt' % (name, name, count), 'wb') as aout:
            pickle.dump(aucdic, aout)
        fout.close()
        tout.close()
        aout.close()
        model_pre.save("./models/%s/%d.model"%(name,count))
        print(aucdic)
    fpr, tpr, aucdic = conf_auc(np.array(all_true_bloodvessel), np.array(all_pred_bloodvessel),lens[num])
    with open('dic_fpr_%s.txt'%(name),'wb') as fout:
        pickle.dump(fpr,fout)
    with open('dic_tpr_%s.txt'%(name),'wb') as tout:
        pickle.dump(tpr,tout)
    with open('dic_auc_%s.txt'%(name),'wb') as aout:
        pickle.dump(aucdic,aout)
    print(aucdic)




