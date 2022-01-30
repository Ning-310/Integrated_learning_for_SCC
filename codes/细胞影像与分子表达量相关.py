from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,BatchNormalization,Activation,Input,Conv2DTranspose,Concatenate,concatenate,UpSampling2D,AveragePooling2D,SeparableConv2D,MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import os
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
from PIL import Image,ImageFile
import pandas as pd
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
def getmodel(input_size,l):
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
os.chdir("/home/wangchenwei/project/gyj/cervical_cancer/clinical/mark/")
set_mooc=pd.read_csv("set_mooc.txt",sep='\t',header=None)
set_mooc=set_mooc.T
set_mooc.columns=['set','mooc']
set_mooc=set_mooc.iloc[1:,:]
set_mooc['type']=set_mooc['set'].apply(lambda x:'_'+x.split('-')[-1])
set_mooc['mooc']=set_mooc['mooc']+set_mooc['type']
set_mooc_dic=dict(zip(set_mooc['mooc'],set_mooc['set']))

expression=pd.read_csv("pro-0612.txt",sep='\t',index_col=1).iloc[:,3:]
expression=expression[expression.index.isin(['PLK1','CDK18','DHCR24'])]
expression=expression.applymap(lambda x:int(x*100) if not np.isnan(x) else x)

outpic=open("../hpv_dic_allpics.txt",'rb')
allpics_mooc=pickle.load(outpic)
negpic=open('../pathology/negative_pics_109.txt','rb')
negpics_mooc=pickle.load(negpic)
w = 128
h = 128
c = 3
for gene in expression.index:
    already=[]
    pics_in = []
    label = []
    for k in allpics_mooc.keys():
        mooc = k.split('~')[0]
        if mooc+'_T' in already:
            continue
        if mooc+'_T' in set_mooc_dic.keys():
            set_num_pos = set_mooc_dic[mooc+'_T']
            set_num_neg = set_mooc_dic[mooc + '_N']
            if not np.isnan(expression.loc[gene, set_num_pos]):
                already.append(mooc+'_T')
                pics_in+=allpics_mooc[k]
                label+=[float(expression.loc[gene, set_num_pos])]*(len(allpics_mooc[k]))
            if not np.isnan(expression.loc[gene, set_num_neg]):
                pics_in+=negpics_mooc[mooc]
                already.append(mooc+'_N')
                label += [float(expression.loc[gene, set_num_neg])] * (len(negpics_mooc[mooc]))
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
        l = y_train.shape[1]
        model = getmodel((w, h, c), l)

        print('----------------marker training-----------------------')
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        checkpoint = ModelCheckpoint('models/%s.model' % (gene), save_weights_only=False, monitor='val_loss',
                                     verbose=1,save_best_only=True, mode='auto', period=1),
        model.fit(x_train, y_train, batch_size=400, epochs=25, validation_data=(x_test, y_test),
                  callbacks=[checkpoint, earlystopping], verbose=1)

        model_pre = load_model('models/%s.model' % (gene))
        x_pred = list(model_pre.predict(x_test))
        with open('./mooc_pic_split/%s_mooc_pred.txt' % (gene), 'a') as sec:
            for loc, mooc in enumerate(list(y_test)):
                sec.write(mooc + '\t')
                for i in x_pred[loc]:
                    sec.write(str(i) + '\t')
                sec.write('\n')