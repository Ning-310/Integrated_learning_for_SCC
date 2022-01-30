# -*- coding: utf-8 -*-
# @Time    : 2021/6/4  10:31
# @Author  : Gou Yujie
# @File    : 4areas_clf_allkinds.py
# -*- coding: utf-8 -*-
# @Time    : 2021/5/31  20:31
# @Author  : Gou Yujie
# @File    : 4areas_clf.py

import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from keras.models import load_model
from PIL import Image,ImageFile
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pylab as pylab
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from tensorflow.keras.metrics import AUC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
import joblib
import cv2
from sklearn.manifold import TSNE
def logistic(data_drug,data_lb):
    data_lb=np.array(data_lb).ravel()
    data_drug=np.array(data_drug)
    lr=LR(penalty='l2',solver='liblinear',C=0.8,class_weight='balanced')
    clf=lr.fit(data_drug,data_lb)
    print(clf.coef_)
    return clf
def roc(y_tests, y_test_scores):
    font = {'family': 'arial',
    'size': 20 }
    params = {'axes.labelsize': '20',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'lines.linewidth': '4'}
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    # pylab.rcParams['font.weight'] = 'bold'
    plt.figure(figsize=(5, 5), dpi=300)
    AUC = roc_auc_score(y_tests, y_test_scores)
    fpr1, tpr1, thresholds1 = roc_curve(y_tests, y_test_scores)
    plt.plot(fpr1, tpr1, linewidth='3', color='tomato', label='Areas Logistic (AUC = {:.3f})'.format(AUC))
    plt.plot([0, 1], [0, 1], linewidth='1', color='grey', linestyle="--")
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(prop={'size': 20}, loc=4, frameon=False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.xlabel('1–Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.savefig('F:\cervical_cancer/all_results_for_reports/4areas_clf_roc.pdf')
    plt.show()
def cnt_area(cnt):
    area=cnt.area
    return area
def get_red_keratin(pic):
    width = pic.size[0]
    height = pic.size[1]
    R, G, B = pic.split()
    R = np.array(R)
    B = np.array(B)
    G = np.array(G)
    thresh = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 211, 11)
    thresh1 = (np.array(thresh) / 255).astype(np.uint8)
    thresh = cv2.adaptiveThreshold(B, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 211, 11)
    thresh2 = (np.array(thresh) / 255).astype(np.uint8)
    thresh = cv2.adaptiveThreshold(R, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 211, 11)
    thresh3 = (np.array(thresh) / 255).astype(np.uint8)
    out = np.array(thresh1 * thresh2 * thresh3)
    G = np.zeros((height, width), np.uint8)
    kernel_all = np.ones((5, 5), np.uint8)
    binary_all = cv2.dilate(out, kernel_all)
    labels = measure.label(binary_all, connectivity=2)
    props = measure.regionprops(labels)
    props.sort(key=cnt_area, reverse=True)
    for k in range(1, len(props)):
        if 250000 > props[k].area > 10000:
            for cor in props[k].coords:
                G[cor[0], cor[1]] = 1
    return G
def find_pro(labpic):
    labpic=np.array(labpic)
    blu = np.array([0, 0, 0])
    dst = cv2.inRange(labpic, blu, blu)
    xy_kera = np.column_stack(np.where(dst==255))
    gre = np.array([0, 255, 0])
    dst = cv2.inRange(labpic, gre, gre)
    xy_gre = np.column_stack(np.where(dst==255))
    return xy_gre, xy_kera
def true_pro(xy_kera,xy_gre):
    true={}
    np.random.seed(count+3)
    locs = np.random.randint(0, len(xy_kera), size=100)
    for i in xy_kera[[k for k in locs],:]:
        if i[0] + 128 <= height and i[1] + 128 <= width:
            true[(i[0], i[1])] = 3
    if len(xy_gre) > 0:
        locs2 = np.random.randint(0, len(xy_gre), size=100)
        for i in xy_gre[[k for k in locs2],:]:
            if i[0] + 128 <= height and i[1] + 128 <= width:
                true[(i[0], i[1])] = 2
    return true
def find_circle_locs(ori_pic,model):
    w = h = 128
    c = 3
    width = ori_pic.size[0]
    height = ori_pic.size[1]
    gray_pic = ori_pic.convert('L')
    gray_pic = np.array(gray_pic)
    ret, thresh1 = cv2.threshold(gray_pic, 245, 255, cv2.THRESH_BINARY)
    dst = 255 - thresh1
    dst = np.array(dst)
    ori_pic = np.array(ori_pic)
    labels2 = measure.label(dst, connectivity=2)
    props2 = measure.regionprops(labels2)
    props_to_minus = []
    for k in range(1, len(props2)):
        if props2[k].area < 23000:
            props_to_minus.append(props2[k])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    binary = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭操作
    binary = np.array(binary)
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)
    props.sort(key=cnt_area, reverse=True)
    props_in_need = []
    for k in range(1, len(props)):
        if 800000 > props[k].area > 40000:
            cor = props[k].centroid
            if 512 < int(cor[0]) < height - 512 and 512 < int(cor[1]) < width - 512:
                img_1024 = ori_pic[int(cor[0]) - 512:int(cor[0]) + 512, int(cor[1]) - 512:int(cor[1]) + 512]
                img_1024 = Image.fromarray(img_1024.astype('uint8')).convert('RGB')
                img_128 = img_1024.resize((w, h), Image.ANTIALIAS)
                img_128 = np.array(img_128) / 255
                score = float(model.predict(img_128.reshape(1, w, h, c))[:, 1])
                if score >= 0.5:
                    props_in_need.append(props[k])
    return props_in_need, props_to_minus
def four_roc(name,can_label,can_pred,mat_label,mat_pred,ves_lab,ves_pred,kera_lab,kera_pred):
    AUC_can = roc_auc_score(can_label, can_pred)
    fpr1, tpr1, thresholds1 = roc_curve(can_label, can_pred)
    AUC_mat = roc_auc_score(mat_label, mat_pred)
    fpr2, tpr2, thresholds2 = roc_curve(mat_label, mat_pred)
    AUC_kera = roc_auc_score(kera_lab, kera_pred)
    fpr3, tpr3, thresholds3 = roc_curve(kera_lab, kera_pred)
    AUC_ves = roc_auc_score(ves_lab, ves_pred)
    fpr4, tpr4, thresholds4 = roc_curve(ves_lab, ves_pred)
    font = {'family': 'arial',
            # 'weight': 'bold',
            'size': 20}
    params = {'axes.labelsize': '20',
              'xtick.labelsize': '20',
              'ytick.labelsize': '20',
              'lines.linewidth': '4'}
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(7, 7), dpi=300)
    plt.plot(fpr1, tpr1, linewidth='3', color='tomato', label='Cancer (AUC = {:.3f})'.format(AUC_can))
    plt.plot(fpr2, tpr2, linewidth='3', color='goldenrod', label='Matrix (AUC = {:.3f})'.format(AUC_mat))
    plt.plot(fpr3, tpr3, linewidth='3', color='steelblue', label='Keratin (AUC = {:.3f})'.format(AUC_kera))
    plt.plot(fpr4, tpr4, linewidth='3', color='seagreen', label='Vessel (AUC = {:.3f})'.format(AUC_ves))
    plt.plot([0, 1], [0, 1], linewidth='1', color='grey', linestyle="--")
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.legend(prop={'size': 20}, loc=4, frameon=False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.xlabel('1–Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.savefig('F:\cervical_cancer/all_results_for_reports/4areas_roc_%s.pdf'%(name))
    plt.show()
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.clim(vmin=0, vmax=1)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    font2 = {'family': 'Arial',
             'size': 20,}
    plt.subplots_adjust(left=0.3, right=0.8, top=0.8, bottom=0.3)
    plt.xticks(rotation=90)
    plt.ylabel('Actual',font2,labelpad = 0)
    plt.xlabel('Predicted',font2,labelpad = 5)
import joblib
import time
import pickle
from sklearn.metrics import confusion_matrix
if __name__=='__main__':

    count=0
    all_true_dic={}
    all_pred_dic={}
    all_score_dic={}
    root = "H:\cancer_area\keratin_new/20201225 origin test/"
    fileroot = "H:/pathology/all_areas_result_new/"
    outfileroot_lab = "H:\cancer_area\keratin_new/labelfile/"
    outfileroot_pred = "H:\cancer_area\keratin_new/predfile/"
    model_circle = load_model("./models/green_circle1.model")
    for file in os.listdir(root):
        print(file)
        loc_area_score = {}
        picpath = os.path.join(root, file)
        labelpath = picpath.replace("20201225 origin test", "keratin_label_test")
        filepath = os.path.join(fileroot, file.strip('.tif') + '_area_all_scores_pred.txt')
        oripic = Image.open(picpath)
        labpic = Image.open(labelpath)
        width = oripic.size[0]
        height = oripic.size[1]
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        pros_hole,props_background=find_circle_locs(oripic,model_circle)
        with open("H:\cancer_area\keratin_new/nes_arrays/pro_hole_%s.txt"%(file.strip('.tif')),'wb') as pr:
            pickle.dump(pros_hole,pr)
        red_pic = get_red_keratin(oripic)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),' get_red_circle')
        labpic = np.array(labpic)
        oripic=np.array(oripic)
        for prop_m in props_background:
            for loc in prop_m.coords:
                oripic[loc[0], loc[1]] = [255, 255, 255]
        xy_gre,xy_kera= find_pro(labpic)  # 在label中标黑色/绿色的部分
        with open("H:\cancer_area\keratin_new/nes_arrays/gre_kera_true_%s.txt"%(file.strip('.tif')),'ab') as gk:
            pickle.dump(xy_gre,gk)
            pickle.dump(xy_kera,gk)
        # true_dic=true_pro(xy_kera,xy_gre)
        with open(filepath, 'r') as txt:
            data = txt.readlines()
            for line in data:
                area_score = line.split('\t')[1].strip().strip("[").strip(']').split(", ")
                area_score = [float(i) for i in area_score]
                area_loc = line.split('\t')[0].strip().split(',')
                area_loc = [int(i.strip('(').strip(')').strip("'")) for i in area_loc]
                area_loc = tuple(area_loc)
                red_part = red_pic[area_loc[0]:area_loc[0] + 128, area_loc[1]:area_loc[1] + 128]
                if (red_part == 0).all():
                    area_score[4] = 0
                part=oripic[area_loc[0]:area_loc[0] + 128, area_loc[1]:area_loc[1] + 128]
                _, gray_G, _ = cv2.split(part)
                if (gray_G > 200).all():
                    continue
                loc_area_score[area_loc] = area_score[1:5]   #所有坐标及其预测结果，以区域为单位
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), ' loc_area_score')
        las=open("H:\cancer_area\keratin_new/nes_arrays/loc_area_score_%s.txt"%(file.strip('.tif')),'wb')
        pickle.dump(loc_area_score,las)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),' get_all_dic')

        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),' true_pred')
        picpath = os.path.join(root, file)
        labelpath = picpath.replace("20201225 origin test", "keratin_label_test")
        labpic = Image.open(labelpath)
        width = labpic.size[0]
        height = labpic.size[1]
        labpic=np.array(labpic)
        with open("H:\cancer_area\keratin_new/nes_arrays/pro_hole_%s.txt" % (file.strip('.tif')), 'rb') as pr:
            pros_hole=pickle.load(pr)
        all_prop=[]
        for prop in pros_hole:
            for loc in prop.coords:
                loc = (loc[0], loc[1])
                all_prop.append(loc)
        all_prop=set(all_prop)

        with open("H:\cancer_area\keratin_new/nes_arrays/gre_kera_true_%s.txt"%(file.strip('.tif')),'rb') as gk:
            xy_gre=pickle.load(gk)
            xy_kera=pickle.load(gk)

        true_dic=true_pro(xy_kera,xy_gre)
        las = open("H:/cancer_area/keratin_new/nes_arrays/loc_area_score_%s.txt" % (file.strip('.tif')), 'rb')
        loc_area_score=pickle.load(las)
        pred_dic = {}
        pred_score_dic={}
        for i2 in true_dic.keys():
            for loc, score in loc_area_score.items():
                if loc[0] <= i2[0] < loc[0] + 128 and loc[1] <= i2[1] < loc[1] + 128:
                    pred_score_dic[i2] = score
                    pred_dic[i2]=score.index(max(score))

        true_dic = {k: v for k, v in true_dic.items() if k in pred_dic.keys()}
        xy_gre=xy_gre.tolist()
        xy_kera=xy_kera.tolist()
        xy_gre_se=((i,j) for [i,j] in xy_gre)
        xy_kera_se=((i,j) for [i,j] in xy_kera)
        np.random.seed(count+1)
        xs = np.random.randint(3000, height-3000, size=100)
        ys = np.random.randint(3000, width-3000, size=100)
        xs, ys = np.meshgrid(xs, ys)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), ' ram_pred')
        for i, j in zip(xs.ravel(), ys.ravel()):
            if (i, j) in set(list(true_dic.keys())):
                pass
            else:
                for loc, score in loc_area_score.items():
                    if (i,j) not in all_prop:
                        score[2] = 0
                    if loc[0] <= i < loc[0] + 128 and loc[1] <= j < loc[1] + 128:
                        pred_score_dic[(i, j)] = score
                        pred_dic[(i, j)] = score.index(max(score))
                        if (i,j) in xy_gre_se:
                            # print(i)
                            true_dic[(i, j)] =2
                            continue
                        elif (i,j) in xy_kera_se:
                            true_dic[(i, j)] = 3
                            continue
                        elif (labpic[i,j]==[255,0,0]).all():
                            # print(i,j)
                            true_dic[(i,j)]=0
                            continue
                        elif (labpic[i,j]==[0,255,255]).all():
                            true_dic[(i,j)]=1
                            continue
                        else:
                            continue
        pred_dic = {k: v for k, v in pred_dic.items() if k in true_dic.keys()}
        pred_score_dic = {k: v for k, v in pred_score_dic.items() if k in true_dic.keys()}
        all_true_dic.update(true_dic)
        all_pred_dic.update(pred_dic)
        all_score_dic.update(pred_score_dic)

        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), ' all_pred')
    with open("H:\cancer_area\keratin_new/nes_arrays/all_true_dic.txt",'wb') as true:
        pickle.dump(all_true_dic,true)
    with open("H:\cancer_area\keratin_new/nes_arrays/all_pred_dic.txt",'wb') as pred:
        pickle.dump(all_pred_dic,pred)
    with open("H:\cancer_area\keratin_new/nes_arrays/all_score_dic.txt",'wb') as score:
        pickle.dump(all_score_dic,score)
    print(len(all_score_dic))

    can_lab=[1 if v==0 else 0 for v in all_true_dic.values()]
    mat_lab = [1 if v == 1 else 0 for v in all_true_dic.values()]
    ves_lab = [1 if v == 2 else 0 for v in all_true_dic.values()]
    kera_lab = [1 if v == 3 else 0 for v in all_true_dic.values()]
    can_pred=[]
    mat_pred=[]
    ves_pred=[]
    kera_pred=[]
    for v in all_score_dic.values():
        print(v)
        can_pred.append(v[0])
        mat_pred.append(v[1])
        ves_pred.append(v[2])
        kera_pred.append(v[3])
    print(len(ves_lab),len(ves_pred))
    four_roc('before',can_lab,can_pred,mat_lab,mat_pred,ves_lab,ves_pred,kera_lab,kera_pred)

    all_train=list(all_score_dic.values())
    all_train=np.array(all_train)
    can_lab=np.array(can_lab)
    mat_lab = np.array(mat_lab)
    ves_lab = np.array(ves_lab)
    kera_lab = np.array(kera_lab)
    np.savetxt("H:\cancer_area\keratin_new\predfile/%s.txt"%('all_train'),all_train,fmt='%.3f')
    np.savetxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('can_lab'), can_lab, fmt='%d')
    np.savetxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('mat_lab'), mat_lab, fmt='%d')
    np.savetxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('ves_lab'), ves_lab, fmt='%d')
    np.savetxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('kera_lab'), kera_lab, fmt='%d')

    with open("H:\cancer_area\keratin_new/nes_arrays/all_true_dic.txt",'rb') as true:
        all_true_dic=pickle.load(true)
    with open("H:\cancer_area\keratin_new/nes_arrays/all_score_dic.txt",'rb') as score:
        all_score_dic=pickle.load(score)
    all_train=np.loadtxt("H:\cancer_area\keratin_new\predfile/%s.txt"%('all_train'))
    can_lab=np.loadtxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('can_lab'))
    mat_lab=np.loadtxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('mat_lab'))
    ves_lab=np.loadtxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('ves_lab'))
    kera_lab=np.loadtxt("H:\cancer_area\keratin_new\labelfile/%s.txt" % ('kera_lab'))
    tsne = TSNE(n_components=2)
    t = tsne.fit_transform(all_train)
    t = np.column_stack((t, np.array(list(all_true_dic.values())).T))
    np.savetxt("F:\cervical_cancer/all_results_for_reports/tsne/area_classify_before.txt", t)

    clf_can = logistic(all_train, can_lab)
    joblib.dump(clf_can,"F:/cervical_cancer/pic_process/models/clf_can.pkl")
    clf_mat = logistic(all_train, mat_lab)
    joblib.dump(clf_mat,"F:/cervical_cancer/pic_process/models/clf_mat.pkl")
    clf_ves = logistic(all_train, ves_lab)
    joblib.dump(clf_ves,"F:/cervical_cancer/pic_process/models/clf_ves.pkl")
    clf_kera = logistic(all_train, kera_lab)
    joblib.dump(clf_kera,"F:/cervical_cancer/pic_process/models/clf_kera.pkl")

    clf_can=joblib.load("F:/cervical_cancer/pic_process/models/clf_can.pkl")
    clf_mat = joblib.load("F:/cervical_cancer/pic_process/models/clf_mat.pkl")
    clf_ves = joblib.load("F:/cervical_cancer/pic_process/models/clf_ves.pkl")
    clf_kera = joblib.load("F:/cervical_cancer/pic_process/models/clf_kera.pkl")
    can_pred_clf=clf_can.predict_proba(all_train)[:,1]
    mat_pred_clf = clf_mat.predict_proba(all_train)[:, 1]
    ves_pred_clf = clf_ves.predict_proba(all_train)[:, 1]*1.31
    kera_pred_clf = clf_kera.predict_proba(all_train)[:, 1]*1.35

    four_roc('After',can_lab,can_pred_clf,mat_lab,mat_pred_clf,ves_lab,ves_pred_clf,kera_lab,kera_pred_clf)

    can_pred_clf = clf_can.predict(all_train)
    mat_pred_clf = clf_mat.predict(all_train)
    ves_pred_clf = clf_ves.predict(all_train)
    kera_pred_clf = clf_kera.predict(all_train)
    labels = ['Cancer','Matrix', 'Keratin', 'Vessel']
    params = {
        'axes.labelsize': '20',
        'xtick.labelsize': '20',
        'ytick.labelsize': '20',
        'lines.linewidth': '4',
        'figure.figsize': '5, 5'
    }
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    true_lab = list(all_true_dic.values())

    true_lab = [4 if i == 2 else i for i in true_lab]
    true_lab = [2 if i == 3 else i for i in true_lab]
    true_lab = [3 if i == 4 else i for i in true_lab]
    out=[]
    all_scores=[]
    all_scores.append(can_pred_clf)
    all_scores.append(mat_pred_clf)
    all_scores.append(kera_pred_clf)
    all_scores.append(ves_pred_clf)
    all_scores=np.array(all_scores)
    print(all_scores)
    tsne = TSNE(n_components=2)
    t = tsne.fit_transform(all_scores.T)
    t = np.column_stack((t, np.array(list(all_true_dic.values())).T))
    print(t)
    np.savetxt("F:\cervical_cancer/all_results_for_reports/tsne/area_classify_after.txt", t)

    for col in range(len(can_pred_clf)):
        data=all_scores[:,col].tolist()
        out.append(data.index(max(data)))
    print(len(out),len(true_lab))
    cm = confusion_matrix(true_lab, out)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=300)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c >= 0.005:
            if c > 0.5:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='w', fontsize=15, va='center', ha='center')
            if c < 0.5:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=15, va='center', ha='center')

    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.2)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.savefig('F:\cervical_cancer/all_results_for_reports/areas_confusion_matrix.pdf')
    plt.show()


