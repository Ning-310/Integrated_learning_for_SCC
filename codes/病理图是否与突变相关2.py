# -*- coding: utf-8 -*-
# @Time    : 2022/1/4  1:28
# @Author  : Gou Yujie
# @File    : 病理图是否与突变相关2.py
# -*- coding: utf-8 -*-
# @Time    : 2022/1/3  23:41
# @Author  : Gou Yujie
# @File    : 病理图与是否突变相关.py
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
from PIL import Image,ImageFile
import pandas as pd
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from sklearn.model_selection import StratifiedKFold
def getmodel(input_size):
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
    dense3=Dense(2,activation='softmax')(drop1)

    model = Model(inputs=[inputs], outputs=[dense3])
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=[AUC(name='auc')])
    print(model.summary())
    return model
set_mooc=pd.read_csv("set_mooc.txt",sep='\t',header=None)
set_mooc=set_mooc.T
set_mooc.columns=['set','mooc']
set_mooc.dropna(how='any',axis=0,inplace=True)
set_mooc['set']=set_mooc['set'].apply(lambda x:x.strip('-N').strip('-T'))
set_mooc.drop_duplicates(keep='first',inplace=True,subset=['mooc'])
set_mooc_dic=dict(zip(set_mooc['set'],set_mooc['mooc']))
outpic=open("../hpv_dic_allpics.txt",'rb')
allpics_mooc=pickle.load(outpic)
basic=open("mutation_info.txt",'r')
w = 128
h = 128
c = 3
for line in basic.readlines()[1:]:
    gene=line.strip().split('\t')[0]
    setnums=[i.strip(' ') for i in line.strip().split('\t')[1:] if i!="''"]
    labels={}
    for setnum in set_mooc_dic.keys():
        if setnum in setnums:
            labels[set_mooc_dic[setnum]]=1
        else:
            labels[set_mooc_dic[setnum]] = 0
    already=[]
    pics=[]
    label=[]
    for k in allpics_mooc.keys():
        mooc = k.split('~')[0]
        if mooc in already:
            continue
        if mooc in set_mooc_dic.values():
            already.append(mooc)
            pics+=allpics_mooc[k]
            label+=[labels[mooc]]*len(allpics_mooc[k])
    print(already,pics,label)

    pics_in = np.array(pics)
    label = np.array(label)
    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    count = 0
    cla = []
    for train, test in sfolder.split(pics_in, label):
        count += 1
        x_train, x_test = pics_in[train], pics_in[test]
        y_train, y_test = label[train], label[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        model = getmodel((w, h, c))
        print('----------------marker training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint('./model_for_mutation/%s_%d.model' % (gene,count), save_weights_only=False, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=400, epochs=25, validation_data=(x_test, y_test),
                  callbacks=[checkpoint, earlystopping], verbose=1)
        model_pre = load_model('./model_for_mutation/%s.model' % (gene))
        x_pred = list(model_pre.predict(x_test))
        with open('./model_for_mutation/%s_mooc_pred.txt' % (gene), 'a') as sec:
            for loc, mooc in enumerate(list(y_test)):
                sec.write(mooc + '\t')
                for i in x_pred[loc]:
                    sec.write(str(i) + '\t')
                sec.write('\n')

