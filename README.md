# HDMC: Hierarchical Distribution Matching and Contrastive learning
Code and data for using HDMC, a novel deep learning based framework for batch effect removal in scRNA-seq data. 

## Install
git clone https://github.com/zhanglabNKU/HDMC.git  
cd HDMC/

## R Dependencies
* Seurat 2.3.0

## Python Dependencies
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
Run the R script pre_processing.R as follows:
```
Rscript pre_processing.R folder_name file1 file2 ...
```
For example:
```
Rscript pre_processing.R example batch1.csv batch2.csv
```
> The two datasets batch1.csv and batch2.csv (must be csv form) will be processed by the script and you will get three files saved in the same folder: the processed data named batch1_seurat.csv and batch2_seurat.csv, a file named metaneighbor.csv containing values of the cluster similarities between different batches.
### Batch correction
Run the python script hdmc.py to combine the datasets and remove batch effects as follows:
```
python hdmc.py -data_folder folder -files file1 file2 ... -h_thr thr1 -l_thr thr2
```
For example:
```
python hdmc.py -data_folder example/ -files batch1_seurat.csv batch2_seurat.csv -h_thr 0.9 -l_thr 0.5
```
> This command will train an HDMC model for the selected files in the data_folder with two thresholds (-h_thr is the higher threshold and -l_thr is the lower one). When the training is finished, the datasets will be combined without batch effectes and the result file named combined.csv will be saved in the same data folder.  

In addition, some optional parameters are also available:
* `-num_epochs`: number of the training epochs (default=2000)
* `-code_dim`: dimension of the embedded code (default=20)
* `-base_lr`: base learning rate for network training (default=1e-3)
* `-lr_step`: step decay of learning rates (default=200)
* `-gamma`: hyperparameter for adversarial learning (default=1)  

Under most circumstances, you don't need to change the optional parameters.  

Use the help command to print all the options:
```
python hdmc.py --help
```

## Data availability
The download links of all the datasets are given in the folder named data.
