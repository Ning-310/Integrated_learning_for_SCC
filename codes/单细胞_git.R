rm(list=ls())
library(dplyr)
library(Seurat)
library(data.table) 
library(DOSE)
library(org.Hs.eg.db)
library(tidyverse)

Transsymbol <- function(gene_list){
  entrez_id<-mapIds(x=org.Hs.eg.db,
                    keys = gene_list,
                    keytype = "UNIPROT",
                    column = "SYMBOL")
  entrez_id <- data.frame(entrez_id)
  return (entrez_id$entrez_id)
}

ori_ct=read.table("J:/BACKUPI/¿Õ¼äµ°°××é/iBAQ_processed_forR.txt",sep='\t',header=T,row.names = 1)
maize <- CreateSeuratObject(counts = ori_ct, 
                            project = "maize")
maize <- NormalizeData(maize, normalization.method = "LogNormalize", 
                       scale.factor = 10000)
maize <- NormalizeData(maize)
maize <- FindVariableFeatures(maize, selection.method = "vst", nfeatures = 1000)
dif <- VariableFeatures(maize)
all.genes <- rownames(maize)
maize <- ScaleData(maize, features = all.genes)
maize <- RunPCA(maize, features = VariableFeatures(object = maize), 
                verbose = FALSE)
maize <- FindNeighbors(maize, dims = 1:10, verbose = FALSE)
maize <- FindClusters(maize, resolution =1.5, verbose = FALSE)
maize <- RunUMAP(maize, dims = 1:10, umap.method = "uwot", metric = "cosine")
dplot <- DimPlot(maize, reduction = "umap", group.by = 'seurat_clusters',
                 label = FALSE, pt.size = 2) 
dplot
