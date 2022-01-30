# -*- coding: utf-8 -*-
# @Time    : 2021/8/9  16:34
# @Author  : Gou Yujie
# @File    : 影像各指标预测结果对病人分型.py
import numpy as np
from PIL import Image,ImageFile
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.cluster import KMeans
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def conf_auc(ground_truth,test_predictions,l):
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(int(l)):
        fpr[i], tpr[i], _ = metrics.roc_curve(ground_truth[:, i], test_predictions[:, i],pos_label=1)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    return fpr,tpr,roc_auc

clinical=pd.read_csv("clinical_data.txt",sep='\t',header=0,index_col=0)
needid=clinical['sample_id'].tolist()
id_stage=dict(zip(clinical['sample_id'],clinical['Stage_2009_adj']))
id_grade=dict(zip(clinical['sample_id'],clinical['Grade']))
id_lym=dict(zip(clinical['sample_id'],clinical['Lym_Meta']))
stages_trans={'IB1':0,'IB2':1,'IIB':2,'IIA1':3,'IIA2':4,'IIIB':5}
for k,v in id_stage.items():
    id_stage[k]=stages_trans[v]
grade_trans={'G1':0,'G1-G2':1,'G2':2,'G2-G3':3,'G3':4}
for k,v in id_grade.items():
    id_grade[k]=grade_trans[v]
for k,v in id_lym.items():
    if v=='Neg':
        id_lym[k]=0
    else:
        id_lym[k] = 1
id_squade=dict(zip(clinical['sample_id'],clinical['Tumor_type']))
for k,v in id_squade.items():
    if v=='squamous':
        id_squade[k]=0
    elif v=='adenocarcinoma':
        id_squade[k]=1
file=pd.read_csv("hpv.txt",sep='\t',header=0,encoding='gbk')
id_clade={}
for k,v in dict(zip(file['patient'],file['RNA.HPV.clade'])).items():
    if v not in [np.nan]:
        if v=='A9':
            id_clade[k.replace('-','.')]=0
        else:
            id_clade[k.replace('-','.')] = 1
id_hpv={}
for k,v in dict(zip(file['patient'],file['hpv'])).items():
    if v not in [np.nan,'nan',np.NAN]:
        id_hpv[k.replace('-','.')]=v-1
file_sub=pd.read_csv('subtype.txt',sep='\t',header=0)
id_subtype=dict(zip(file_sub['sample_id'],file_sub['subt']))
file2=pd.read_csv('clinical_data.txt',sep='\t',header=0)
id_grade_2018=dict(zip(file2['sample_id'],file2['Stage_FIGO2018']))
trans_dic={'IIIB':0,'IIB':1,'IIIC1r':2,'IIA2':3,'IIA1':4,'IB1':5,'IB2':6,'IB3':7,'IIIC1p':8}
for k,v in id_grade_2018.items():
    id_grade_2018[k]=trans_dic[v]

outpic=open("hpv_dic_allpics.txt",'rb')
allpics_mooc=pickle.load(outpic)
keys=['stages','grades','grade_2018','lyms','clade','hpv','subtype']
lens=[6,5,9,2,2,3,3]
dics=[id_stage,id_grade,id_grade_2018,id_lym,id_clade,id_hpv,id_subtype]
for loc,name in enumerate(keys):
    pred_out=[]
    with open('models/%s/over_predsdic_%s.txt' % (name, name), 'rb') as tout:
        preds = pickle.load(tout)
        for v in preds.values():
            v = np.array(v)
            for k in range(int(v.shape[0] / 5)):
                resu = v[k * 5:(k + 1) * 5, :]
                pred_out.append(np.mean(resu,axis=0))
    pred_path=[]
    for k in allpics_mooc.keys():
        mooc = k.split('~')[0]
        if mooc in dics[loc].keys():
            pred_path.append(k)
    path_out=dict(zip(pred_path,pred_out))
    with open('models/%s/path_pred_%s.txt' % (name, name), 'wb') as rout:
        pickle.dump(path_out,rout)
    rout.close()

def chart_for_chi(list1,list2):
    chart=np.zeros((len(set(list1)),len(set(list2))))
    for loc, i in enumerate(list1):
        chart[i, list2[loc]] += 1
    return chart
protein_subtype=pd.read_csv("H:\pathology_clinical/all_kinds_subtypes.txt",sep='\t',header=0)
pro_sub_dic=dict(zip(protein_subtype['sample_id'],protein_subtype['subt']))
clade_dic=dict(zip(protein_subtype['patient'],protein_subtype['RNA.HPV.clade']))
for k,v in clade_dic.items():
    if v not in [np.nan]:
        if v=='A9':
            clade_dic[k]=1
        else:
            clade_dic[k] = 2
stage_dic=dict(zip(protein_subtype['sample_id2'],protein_subtype['Stage_2009_adj']))
stages_trans={'IB1':0,'IB2':1,'IIB':2,'IIA1':3,'IIA2':4,'IIIB':5}
for k,v in stage_dic.items():
    stage_dic[k]=stages_trans[v]

keys=['stages','grades','grade_2018','lyms','subtype','clade','hpv']
lens=[6,5,9,2,3,3,2]
whole={}
hout=open("path_pred_%s.txt" % ('hpv'), 'rb')
hpv_dic=pickle.load(hout)
hout.close()
cout=open("path_pred_%s.txt" % ('clade'), 'rb')
clade_dic=pickle.load(cout)
cout.close()
with open("path_pred_%s.txt" % ('stages'), 'rb') as sout:
    stages_dic = pickle.load(sout)
    for path in stages_dic.keys():
        whole[path]=[]
for loc,name in enumerate(keys):
    with open("path_pred_%s.txt"%(name),'rb') as sout:
        path_dic=pickle.load(sout)
        for path in path_dic.keys():
            if path in whole.keys():
                whole[path]+=list(path_dic[path])
hpv_mean=np.mean(np.array([i for i in hpv_dic.values()]),axis=0)
clade_mean=np.mean(np.array([i for i in clade_dic.values()]),axis=0)

for path,v in whole.items():
    if len(v)<30:
        whole[path]+=list(clade_mean)
        whole[path]+=list(hpv_mean)
mooc_pred={}
for k,v in whole.items():
    mooc=k.split('~')[0]
    mooc_pred[mooc]=[v]
    if mooc in mooc_pred.keys():
        mooc_pred[mooc].append(v)
whole_km=[]
for key,i in mooc_pred.items():
    whole_km.append(list(np.mean(np.array(i),axis=0)))
whole_km=np.array(whole_km)
kmeans=KMeans(n_clusters=3,random_state=0)
kmeans.fit(whole_km)
y_pred=kmeans.predict(whole_km)

