# -*- coding: utf-8 -*-
# @Time    : 2020/12/30 18:53
# @Author  : Gou Yujie
# @File    : can_beside.py


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC
from keras.models import load_model
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint,EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def model_env(input_size):
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

    conv5 = Dropout(0.5)(pool4)
    f1 = Flatten()(conv5)
    dense1 = Dense(4096, activation='relu')(f1)
    bn1 = BatchNormalization()(dense1)
    ac1 = Activation('relu')(bn1)
    drop1 = Dropout(0.4)(ac1)
    dense3 = Dense(3, activation='softmax')(drop1)

    model = Model(inputs=[inputs], outputs=[dense3])
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=[AUC(name='auc')])
    print(model.summary())
    return model

def load_data():
    images=[]
    types=[]
    f=r"H:/pathology/"
    folders=[f+'cancer_to_train/',f+'next_to_cancer/']
    for i,fo in enumerate(folders):
        for root, dirs, files in os.walk(fo):
            for file in files:
                imgpath = os.path.join(root,file)
                img=Image.open(imgpath)
                img = np.array(img)
                images.append(img)
                types.append(i)
    images=np.array(images)
    types=np.array(types)
    return images,types


def classify(data):
    predicts = []
    pred_for_tsne=[]
    count=0
    for image in data:
        image = np.uint8(image.reshape(1, w, h, c))
        count += 1
        pred = model.predict(image,batch_size=10).tolist()
        layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        pred_out = layer_model.predict(image, batch_size=10).tolist()
        pred_for_tsne.append(pred_out[0])
        predicts.append(pred[0])
    return predicts,pred_for_tsne

import matplotlib.pylab as pylab
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import os

def roc(true,pred,count):
    font = {'family': 'arial',
            'size': 20}
    params = {'axes.labelsize': '20',
              'xtick.labelsize': '20',
              'ytick.labelsize': '20',
              'lines.linewidth': '4'}
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(5, 5), dpi=300)
    AUC = roc_auc_score(true, pred)
    fpr1, tpr1, thresholds1 = roc_curve(true, pred)
    plt.plot(fpr1, tpr1, linewidth='3', color='tomato', label='Quality Control \n (AUC = {:.3f})'.format(AUC))
    plt.plot([0, 1], [0, 1], linewidth='1', color='grey', linestyle="--")
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(prop={'size': 20}, loc=4, frameon=False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.xlabel('1–Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.savefig('F:\cervical_cancer/all_results_for_reports/cb_5folds_roc_%d.pdf'%(count))
    plt.show()

import pickle

if __name__=='__main__':
    w = h = 256
    c = 3
    images,types=load_data()
    model=model_env((w,h,c))
    all_preds=[]
    all_labs=[]
    all_loc_pred={}
    all_loc_label={}
    count=0
    count_auc={}
    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for train, test in sfolder.split(images, types):
        count+=1
        x_train, x_test =images[train],images[test]
        y_train, y_test =types[train],types[test]
        x_train=np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        print('----------------green_bloodvessel training-----------------------')
        earlystopping=EarlyStopping(monitor='val_loss',patience=10,mode='auto')
        checkpoint = ModelCheckpoint('F:\cervical_cancer\pic_process\cancer_beside_models/classify_cb_%d.model'%(count), save_weights_only=False, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='auto')
        model.fit(x_train, y_train, batch_size=64, epochs=50,validation_data=(x_test, y_test),callbacks=[checkpoint,earlystopping],
                  verbose=1)
        model_pre=load_model('F:\cervical_cancer\pic_process\cancer_beside_models/classify_cb_%d.model'%(count))
        x_pred=model_pre.predict(x_test)[:,1]
        loc_pred=dict(zip(test,list(x_pred)))
        loc_label=dict(zip(test,list(y_test[:,1])))

        model_pre.save("F:\cervical_cancer/pic_process/cancer_beside_models/classify_cb_%d.model"%(count))

        with open("F:\cervical_cancer/pic_process/cancer_beside_models/4folds_dic_pred_label_cb_%s.txt"%(count),'wb',-1) as one:
            pickle.dump(loc_pred, one)
            pickle.dump(loc_label,one)
            one.close()
        all_loc_pred.update(loc_pred)
        all_loc_label.update(loc_label)
        AUC1 = roc_auc_score(np.array(y_test[:,1]).reshape(-1, 1), np.array(x_pred).reshape(-1, 1))
        count_auc[count]=AUC1
        print(AUC1)
        roc(np.array(y_test[:,1]).reshape(-1, 1), np.array(x_pred).reshape(-1, 1),count)
    all_truth=[]
    all_pred=[]
    for i in range(1,11):
        with open("cancer_beside_models/4folds_dic_pred_label_cb_%d.txt"%(int(i)),'rb') as out:
            all_loc_pred=pickle.load(out)
            all_loc_label=pickle.load(out)
            print(all_loc_pred,all_loc_label)
            out.close()
        for k,v in  all_loc_label.items():
            all_truth.append(v)
            all_pred.append(all_loc_pred[k])
    roc(np.array(all_truth).reshape(-1, 1), np.array(all_pred).reshape(-1, 1),'all')

