# HDMC: Hierarchical Distribution Matching and Contrastive learning
Code and data for using HDMC, a novel deep learning based framework for batch effect removal in scRNA-seq data. 

## Install
git clone https://github.com/zhanglabNKU/HDMC.git  
cd HDMC/

## R Dependencies
* Seurat 2.3.0

## python Dependencies
* Python 3.7.7
* scikit-learn 0.23.2
* pytorch 1.3.1
* imbalanced-learn 0.7.0
* rpy2 2.9.4
* universal-divergence 0.2.0
* pandas 1.0.4

## Usage
Given several datasets (each treated as a batch) for combination, there are two main steps: (i) preprocess the datasets and run metaneighbor algorithm to compute cluster similarities; (ii) train an HDMC model for batch correction.
### Data preprocessing
* Run the R script pre_processing.R as follows:
```
Rscript pre_processing.R folder_name file1 file2 ...
```
For example:
```
Rscript pre_processing.R example batch1.csv batch2.csv
```
