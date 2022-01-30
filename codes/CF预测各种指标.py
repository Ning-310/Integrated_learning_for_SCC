# -*- coding: utf-8 -*-
# @Time    : 2021/9/15  20:17
# @Author  : Gou Yujie
# @File    : CF预测各种指标.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
import tensorflow as tf
import matplotlib.pylab as pylab
from sklearn.preprocessing import StandardScaler
tf.compat.v1.disable_eager_execution()
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,roc_auc_score
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency

def logistic(data_drug,data_lb):

    data_lb=np.array(data_lb)
    data_drug=np.array(data_drug)
    lr=LR(penalty='l2',solver='liblinear')
    clf=lr.fit(data_drug,data_lb)
    return clf
def conf_auc(ground_truth,test_predictions,l):
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(int(l)):
        fpr[i], tpr[i], _ = metrics.roc_curve(ground_truth[:, i], test_predictions[:, i],pos_label=1)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    return fpr,tpr,roc_auc

def plot_confusion_matrix(y_true,y_pred,labels,title,cmap=plt.cm.Blues):
    params = {
        'axes.labelsize': '10',
        'xtick.labelsize': '5',
        'ytick.labelsize': '5',
        'lines.linewidth': '4',
        'figure.figsize': '3, 3'
    }
    plt.rcParams.update(params)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 9), dpi=300)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    # print(cm_normalized)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c >= 0.005:
            if c >= 0.5:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='w', fontsize=20, va='center', ha='center')
            if c < 0.5:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=20, va='center', ha='center')

    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.2)

    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.clim(vmin=0, vmax=1)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    font2 = {'family': 'Arial',
             'size':10,}
    plt.subplots_adjust(left=0.3, right=0.8, top=0.8, bottom=0.3)
    plt.xticks(rotation=90)
    plt.ylabel('Actual',font2,labelpad = 0)
    plt.xlabel('Predicted',font2,labelpad = -1)
    plt.savefig('H:\pathology_clinical\CF预测指标\matrix/%s.pdf'%(title))
    plt.savefig('H:\pathology_clinical\CF预测指标\matrix/%s.jpg' % (title))
    plt.show()

def multi_roc(true_arr,pred_arr,i):
    fpr, tpr, aucdic = conf_auc(true_arr, pred_arr, nums[i])
    print(fpr,tpr)
    font = {'family': 'arial',
            'size': 20}
    params = {'axes.labelsize': '20',
              'xtick.labelsize': '20',
              'ytick.labelsize': '20',
              'lines.linewidth': '4'}
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(7, 7), dpi=300)
    stages_dict = dics[i]  # hpv专属
    st = list(stages_dict.keys())[0]

    if nums[i] > 2:
        for i in range(len(stages_dict)):
            st = list(stages_dict.keys())[i]
            print(st)
            plt.plot(fpr[i], tpr[i], linewidth='3', color=color_dic[i],
                     label='%s (AUC = %.3f)' % (st, metrics.auc(fpr[i], tpr[i])))
    else:
        print(stages_dict, nums[i])
        plt.plot(fpr[0], tpr[0], linewidth='3', color=color_dic[0],
                 label='%s (AUC = %.3f)' % (list(stages_dict.keys())[0], metrics.auc(fpr[0], tpr[0])))
    plt.plot([0, 1], [0, 1], linewidth='1', color='gray', linestyle="--")
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.legend(prop={'size': 15}, loc=4, frameon=False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.xlabel('1–Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.savefig('H:\pathology_clinical\CF预测指标/roc/%s.jpg' % (col))
    plt.savefig('H:\pathology_clinical\CF预测指标/roc/%s.pdf' % (col))
    plt.show()

id_pathology={0:0,1:1,2:2}
id_subt={0:0,1:1,2:2}
id_clade={0:'A9',1:'Others'}
id_stage={'IB1':0,'IB2':1,'IIA1':2,'IIA2':3,'IIB':4,'IIIB':5}
id_grade={'G1':0,'G2':1,'G3':2}
id_hpv={0:'hpv16',1:'hpv18',2:'Others'}
grade_2018_dic={'IIIB':0,'IIB':1,'IIIC1r':2,'IIA2':3,'IIA1':4,'IB1':5,'IB2':6,'IB3':7,'IIIC1p':8}
id_lym={'Neg':0,'Pos':1}
cfs=pd.read_csv("H:\pathology_clinical/cf_for_calculate.txt",sep='\t',index_col=0,encoding='gbk')
namelist=['id_pathology','id_subt','id_clade','id_hpv','id_stage','id_stage_2018','id_lym','id_grade']

conti_cf=['WBC (10^9/L) ', 'RBC (10^12/L) ','Neutrophil  (10^9/L) ', 'Lymphocyte (10^9/L) ',
       'Cell proportion (%) ', 'Eosinophil(10^9/L) ','urine specific gravity','ALB/GLB','UREA (mmol/L) ', 'creatinine (μmol/L) ','cholesterin (mmol/L) ', 'TG (mmol/L) ',
          'LDL(mmol/L) ','HDL (mmol/L) ', 'PT(s) ', 'INR', 'APTT(s) ', 'TT(s) ', ' FIB(g/L) ',
       'D-dimer(mg/L) ','blood glucose(mmol/L) ', ' SCCA (ng/ml) ',
       'CA-125(U/ml) ', 'CEA(ng/ml) ', 'CA-153(U/ml) ', 'CA-199(U/ml) ', 'HE4','PLT (10^9/L) ','HB (g/L) ','ALT(U/L) ', 'AST(U/L) ', 'Ki-67(%) '
       ]
dics=[id_pathology,id_subt,id_clade,id_hpv,id_stage,grade_2018_dic,id_lym,id_grade]

parts_cf=['urine protein','urine glucose',' CK5/6 -,+,+,++,++', ' CK7 -,+,+,++,++', 'P53 -,+,+,++,++',
       'P63 -,+,+,++,++', 'P+6 -,+,+,++,++', 'CEA -,+,+,++,++',' CD+ -,+,+,++,++', 'CD34 -,+,+,++,++', 'ERG\n -,+']
nums=[3,3,2,3,6,9,2,5]

color_dic={0:'tomato',1:'goldenrod',2:'steelblue',3:'seagreen',4:'#a24857',5:'#8B008B',6:'#6495ED',7:'#008000',8:'#DAA520'}
def RGB_to_Hex(RGB):
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color
color_dic={0:[153,234,253],1:[154,255,100],2:[255,229,13],3:[255,153,51],4:[204,77,38],5:[127,1,0]}
for k,v in color_dic.items():
    color_dic[k]=RGB_to_Hex(v)
print(len(conti_cf)+len(parts_cf))

'''
逻辑回归
'''
cfs.replace({'+': 1, '-': 0,'NA':np.nan},inplace=True)
font = {'family': 'arial',
        'size': 20}
for i,col in enumerate(namelist):
    print(col)
    frame=cfs[cfs[col].notnull()]
    all_train=frame[conti_cf+parts_cf].astype(float)
    for cf in all_train.columns:
        all_train[cf]=all_train[cf].fillna(np.nanmean(all_train[cf]))
    all_train=np.array(all_train)
    all_label=list(frame[col].astype(float))

    for loc, k in enumerate(all_label):
        if k == 0 or k == 1:
            all_label[loc] = 0
        elif k == 3 or k == 4:
            all_label[loc] = 2
        else:
            all_label[loc] = 1

    all_label=np.array(all_label)
    kfold = KFold(n_splits=10, shuffle=True, random_state=5)
    count=0
    all_prediction=np.array([0]*nums[i])
    all_truelables=np.array([0]*nums[i])
    for train, test in kfold.split(all_train, all_label):
        count+=1
        all_train=np.array(all_train)
        all_label=np.array(all_label)
        x_train, x_test = all_train[train], all_train[test]
        y_train, y_test = all_label[train], all_label[test]
        sc=StandardScaler()
        sc.fit(x_train)
        x_train_std=sc.transform(x_train)
        x_test_std = sc.transform(x_test)
        classifier=logistic(x_train_std,y_train)
        predictions = classifier.predict_proba(x_test_std)
        y_test_fla = np.array(to_categorical(y_test, nums[i]))
        all_prediction=np.vstack((all_prediction,predictions))
        all_truelables = np.vstack((all_truelables, y_test_fla))
    all_truelables=np.array(all_truelables)
    all_prediction = np.array(all_prediction)
    for i in range(nums[i]):
        AUC = roc_auc_score(all_truelables[:,i], all_prediction[:,i])
        fpr1, tpr1, thresholds1 = roc_curve(all_truelables[:,i], all_prediction[:,i])
        plt.plot(fpr1, tpr1, linewidth='3', color='tomato', label='AUC = {:.3f}'.format(AUC))
    plt.plot([0, 1], [0, 1], linewidth='1', color='gray', linestyle="--")
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(prop={'size': 20}, loc=4, frameon=False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.xlabel('1–Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.savefig('H:\pathology_clinical\CF预测指标\models/%s_roc.jpg'%(col))
    plt.show()

    all_truelables=all_truelables[1:,:]
    all_prediction = all_prediction[1:, :]
    label =list(dics[i].keys())
    print(all_truelables,all_prediction)

    y_test = []
    predicts = []
    for i in range(all_truelables.shape[0]):
        y_test.append(list(all_truelables[i,:]).index(1))
        pre=list(all_prediction[i,:])
        predicts.append(pre.index(max(pre)))
    print(y_test)
    print(predicts)
    plot_confusion_matrix(y_test,predicts,label,col,cmap=plt.cm.Blues)

def BH(data):
    rk=data.rank()
    num=data.shape[0]
    qvalue=[i*num/list(rk)[loc] for loc,i in enumerate(data)]
    return qvalue
def crosschart(list1,list2):
    frame=pd.DataFrame()
    frame['list1']=list1
    frame['list2']=list2
    return pd.crosstab(frame['list1'],frame['list2'])
basic=pd.read_csv("H:\pathology_clinical/basicinfos.txt",sep='\t',index_col=2,encoding='gbk')
cf_before=basic[basic.columns[20:31]]
cf_before.replace({'+': 1, '-': 0,'NA':np.nan},inplace=True)
cfs.index=cfs['name']
subtype=cfs[namelist]
cfs=cfs[cfs.columns[1:-11]]
total_cf=pd.concat([cf_before,cfs],axis=1)
total_cf=pd.concat([total_cf,subtype],axis=1)
outall=pd.DataFrame()
for loc,kind in enumerate(namelist):
    pval_dic = {}
    num=nums[loc]
    for cf in total_cf.columns[:-9]:
        need=total_cf[[cf,kind]]
        need.dropna(axis=0,inplace=True,how='any')
        # print(need)
        if cf in conti_cf:
            dat=str()
            for i in range(num):
                dat+="need[need['{}']=={:.1f}][cf].tolist(),".format(kind,i)
            code="kruskal("+dat.strip(',')+')'
            pval_dic[cf] = eval(code)[1]

        else:
            pval_dic[cf]=chi2_contingency(crosschart(need[cf],need[kind]))[1]
    pval_dic={k.replace('\n',''):v for k,v in pval_dic.items()}
    pval_dic=pd.DataFrame.from_dict(pval_dic,orient='index',columns=['pval_%s'%(kind)])
    pval_dic['pval_%s'%(kind)]=BH(pval_dic['pval_%s'%(kind)])
    pval_dic[kind]=pval_dic.index
    outall=pd.concat([outall,pval_dic],axis=1)
get=[]
for col in outall.columns:
    if col.startswith('pval_'):
        get.append(col)
outall=outall[get]
print(outall)
outall.to_csv("H:\pathology_clinical\CF预测指标/cf_pval.txt",sep='\t')




