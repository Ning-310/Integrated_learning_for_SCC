# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 14:56
# @Author  : Gou Yujie
# @File    : can_side.py

# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 10:31
# @Author  : Gou Yujie
# @File    : recog.py

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,MaxPooling2D
from keras.utils import to_categorical
from sklearn import metrics
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint,EarlyStopping
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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

    conv5 = Dropout(0.5)(pool4)
    f1=Flatten()(conv5)
    dense1 = Dense(4096, activation='relu')(f1)
    bn1 = BatchNormalization()(dense1)
    ac1 = Activation('relu')(bn1)
    drop1 = Dropout(0.4)(ac1)
    dense3=Dense(3,activation='softmax')(drop1)

    model = Model(inputs=[inputs], outputs=[dense3])
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=[AUC(name='auc')])
    print(model.summary())
    return model

def load_data():
    images=[]
    types=[]
    f=r"H:\pathology\recog/"
    folders=[f+'cell_cancer/',f+'cell_immune/',f+'cell_other/']
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
    types = to_categorical(types)
    return images,types

def classify(data):
    predicts = []
    for image in data:
        image = np.uint8(image.reshape(1, w, h, c))
        pred = model.predict(image,batch_size=10)
        predicts.append(pred)
    return predicts

def conf_auc(test_predictions, ground_truth, confint=0.95):
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(2):
        fpr[i], tpr[i], _ = metrics.roc_curve(ground_truth[:, i], test_predictions[:, i],pos_label=1)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    print(fpr,tpr,roc_auc)
    plt.figure(figsize=(6, 6))
    plt.title('classify_can_imu ROC')
    lables = ['cancer', 'next']
    colors = ['red', 'blue']
    for k,label in enumerate(lables):
        plt.plot(fpr[k], tpr[k], color=colors[k], label=label+'(Val AUC = %0.3f)' % roc_auc[k])
    plt.legend(loc='lower right')
    font = {'color': 'red',
            'size': 20,
            'family': 'Times New Roman'}
    plt.text(0.6, 0.13, '{:0.0f}% confidence interval'.format(confint * 100, fontdict=font))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__=='__main__':
    w = h = 64
    c = 3
    images, types = load_data()
    count=0
    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for train, test in sfolder.split(images, types):
        count += 1
        x_train, x_test = images[train], images[test]
        y_train, y_test = types[train], types[test]
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))
        model=model((w,h,c))
        print('----------------training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint(f'models/classify_filt_canimu_%d.model'%(count), save_weights_only=False, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=200, epochs=50, validation_data=(x_test, y_test),callbacks=[checkpoint,earlystopping], verbose=1,class_weight={0:1288878500,1:1733839016,2:174060250})

        predicts=classify(x_test)
        pre=[]
        for i in predicts:
            pre.append(i.flatten())
        pre=np.array(pre)
        print(pre,y_test)
        conf_auc(pre, y_test,confint=0.95)

