# -*- coding: utf-8 -*-
# @Time    : 2021/5/12  15:39
# @Author  : Gou Yujie
# @File    : all_pats_degree_of_infiltration_linux.py
# -*- coding: utf-8 -*-
# @Time    : 2021/5/12  10:33
# @Author  : Gou Yujie
# @File    : all_pats_degree_of_infiltration.py
import numpy as np
import os
import pickle
from skimage import measure
import cv2
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from keras.models import load_model
import re
def log_mean_exp(a):
    max_ = a.max(axis=1)
    return max_ + np.log(np.exp(a - np.expand_dims(max_, axis=0)).mean(1))
def gaussian_window(x, mu, sigma):
    a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / sigma
    b = np.sum(- 0.5 * (a ** 2), axis=-1)
    E = log_mean_exp(b)
    Z = mu.shape[1] * np.log(sigma * np.sqrt(np.pi * 2))
    return np.exp(E - Z)

def cancer_infiltration(loc_list):
    all_cancer_impacts = 0
    sigma1 = 56
    r = 128
    count=0
    for xy in loc_list:
        x = xy[0]
        y = xy[1]
        if x<=h_pic-r and y<=w_pic-r:
            part = img[x:x + r, y:y + r]
            _, gray_G, _ = cv2.split(part)
            if (gray_G > 200).all():
                continue
            xs = np.arange(x, x + r, 16)
            ys = np.arange(y, y + r, 16)
            xs, ys = np.meshgrid(xs, ys)
            count+=1
            for i, j in zip(xs.ravel(), ys.ravel()):
                all_cancer_impacts+=float(gaussian_window([[i, j]], cancers, sigma1))
    cancer_inf=all_cancer_impacts/(count*128*128)
    return all_cancer_impacts,cancer_inf,count

def imun_infiltration():
    sigma=82
    all_imun_impacts=0
    if len(imun)==0:
        pass
    else:
        for i, j in zip(cancers[:,0].ravel(), cancers[:,1].ravel()):
            all_imun_impacts+=gaussian_window([[i, j]], imun, sigma)
    return all_imun_impacts
def cnt_area(cnt):
    area=cnt.area
    return area

def find_circle_locs(ori_pic,model):
    w=h=128
    c=3
    gray_pic=ori_pic.convert('L')
    gray_pic=np.array(gray_pic)
    ret, thresh1 = cv2.threshold(gray_pic,245, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    binary = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel,iterations=2)
    binary = np.array(binary)
    labels = measure.label(binary,connectivity=2)
    props = measure.regionprops(labels)
    props.sort(key=cnt_area,reverse=True)
    props_in_need=[]
    for k in range(1,len(props)):
        if 800000>props[k].area>40000:
            cor=props[k].centroid
            if 512<int(cor[0])<w_pic-512 and 512<int(cor[1])<h_pic-512:
                img_1024=img[int(cor[0])-512:int(cor[0])+512,int(cor[1])-512:int(cor[1])+512]
                img_1024=Image.fromarray(img_1024.astype('uint8')).convert('RGB')
                img_128 = img_1024.resize((w, h), Image.ANTIALIAS)
                img_128 = np.array(img_128)/255
                score=float(model.predict(img_128.reshape(1,w,h,c))[:,1])
                if score>=0.5:
                    props_in_need.append(props[k])
    return props_in_need

def vessel_infiltration(props):
    vessel_inf=0
    sigma=56
    all_prop=[]
    if len(props)==0:
        pass
    else:
        for prop in props:
            for loc in prop.coords:
                loc=[loc[0],loc[1]]
                all_prop.append(loc)
        all_prop=np.array(all_prop)
        for i, j in zip(cancers[:, 0].ravel(), cancers[:, 1].ravel()):
            vessel_inf += gaussian_window([[i, j]], all_prop, sigma)
    return vessel_inf
c = 3
model_circle = load_model("./green_circle1.model")
file_fo="./all_cancer_imun/"
areas_fo="../process_areas/can_mat_blood_keratin_predict/all_areas_result2/"
pats_cancer_inf={}
pats_imue_inf={}
pats_vessel_inf={}
cancer_out=open("./all_inf2/cancer_inf.txt",'a')
imue_out=open("./all_inf2/imun_inf.txt",'a')
vessel_out=open("./all_inf2/vessel_inf.txt",'a')
can=open("./all_inf2/cancer_inf.txt",'r')
data=can.readlines()
for i in range(1,16):
    picfo="../pathology/f%s"%str(i)+'_original/'
    for pic in os.listdir(picfo):
        if pic.endswith('.tif'):
            flag = 0
            for line in data:
                if re.findall(pic.replace('.tif',''),line):
                    flag=1
                    break
            if flag==0:
                try:
                    ori_pic=Image.open(os.path.join(picfo,pic))
                except:
                    continue
                w_pic = ori_pic.size[0]
                h_pic = ori_pic.size[1]
                img=np.array(ori_pic)
                cancers_p = open(
                    os.path.join(file_fo, pic.replace('.tif','') + '_score_cancer.txt'),
                    'rb')
                imun_p = open(os.path.join(file_fo, pic.replace('.tif','') + '_score_imue.txt'), 'rb')
                cancers = pickle.load(cancers_p)
                imun = pickle.load(imun_p)
                areas=os.path.join(areas_fo,pic.replace('.tif','')+'_area_all_scores_pred.txt')
                loc_area=[]
                with open(areas,'r') as d:
                    data=d.readlines()
                    for line in data:
                        loc = line.split('\t')[0].strip().split(',')
                        loc = [int(i.strip('(').strip(')').strip("'")) for i in loc]
                        loc_area.append(loc)
                key = pic.replace('.tif','')


                print(loc_area)
                all_imps,cancer_inf,count=cancer_infiltration(loc_area)
                print(cancer_inf)
                cancer_out.write(str(key) + '\t' + str(all_imps)+ '\t' + str(cancer_inf)+'\t'+str(count) + '\n')
                cancer_out.flush()

                props=find_circle_locs(ori_pic,model_circle)
                vessel_inf=vessel_infiltration(props)
                print(vessel_inf)
                vessel_out.write(str(key) + '\t' + str(vessel_inf) + '\n')
                vessel_out.flush()

                imue_inf=imun_infiltration()
                imue_out.write(str(key) + '\t' + str(imue_inf) + '\n')
                imue_out.flush()

                pats_cancer_inf[key]=cancer_inf
                pats_imue_inf[key]=imue_inf
                pats_vessel_inf[key]=vessel_inf
    cancer_out=open("./all_inf/cancer_inf.txt",'a')
    imue_out=open("./all_inf/imun_inf.txt",'a')
    vessel_out=open("./all_inf/vessel_inf.txt",'a')
    for k,v in pats_cancer_inf:
        cancer_out.write(str(k)+'\t'+str(v))
    for k,v in pats_imue_inf:
        imue_out.write(str(k)+'\t'+str(v))
    for k,v in pats_vessel_inf:
        vessel_out.write(str(k)+'\t'+str(v))
