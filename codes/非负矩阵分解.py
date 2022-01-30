# -*- coding: utf-8 -*-
# @Time    : 2022/1/14  15:33
# @Author  : Gou Yujie
# @File    : 非负矩阵分解.py

import numpy as np
import pandas as pd
def matrix_factorisation(R, P, Q, K, steps=500, alpha=0.01, beta=0.002):
    Q = Q.T
    for step in range(steps):
        print(step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        print(e)
        if e < 0.001:
            break
    P[P<0]=0
    return P
import re
bulk_prot=pd.read_csv("H:\pathology_biomarkers\分子表达\pro-0612.txt",sep='\t',index_col=0)
bulk_prot=bulk_prot[bulk_prot.index.notnull()]
bulk_prot.dropna(subset=['Gene name'],inplace=True)
genename_dic=dict(zip(bulk_prot.index,bulk_prot['Gene name']))

ibaq = pd.read_csv("J:\BACKUPI\空间蛋白组\iBAQ.txt", sep='\t', index_col=0)
inds=[]
for i in ibaq.index:
    if i in genename_dic.keys():
        inds.append(genename_dic[i])
    else:
        inds.append(np.nan)
ibaq.index=inds
ibaq=ibaq[ibaq.index.notnull()]
prot=pd.read_csv("J:\BACKUPI\空间蛋白组\onlycancer_3cluster_markers.txt",sep='\t',index_col=0)
protein_important=list(prot['protein_name'])
clust=pd.read_csv("J:\BACKUPI\单细胞信息new\每块细胞类型及类群_T_com.txt",sep='\t',index_col=0,encoding='gbk')
clust.fillna(0,inplace=True)
protein_ok=[]
for prot in protein_important:
    if prot in ibaq.index:
        protein_ok.append(prot)
others=ibaq[~ibaq.index.isin(protein_ok)]
need=others.index
outneed=pd.DataFrame()
for pro in protein_ok:
    outneed=pd.concat([outneed,pd.DataFrame(ibaq.loc[pro]).T])
expression=np.array(outneed)
others=np.array(others)
N=expression.shape[0]
K=18
M=expression.shape[1]
P = np.random.rand(N,K)
clust_express=matrix_factorisation(expression,P,others,K)
clust_express=pd.DataFrame(clust_express,index=need,columns=list(clust.columns))
clust_express.to_csv("J:\BACKUPI\单细胞信息new/clust_other_matrixfactorisation.txt",sep='\t')
