#!/usr/bin/env python
import torch.utils.data
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from utils.adversarial import training, testing
from utils.multi_adversarial import multi_training
from utils.pre_processing import pre_processing, read_cluster_similarity
import argparse


# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

# IMPORTANT PARAMETER
similarity_thr = 0.9  # S_thr in the paper, choose between 0.85-0.9

# nn parameter
code_dim = 20
batch_size = 50  # batch size for each cluster
base_lr = 1e-3
lr_step = 200  # step decay of learning rates
momentum = 0.9
l2_decay = 5e-5
gamma = 1  # regularization between reconstruction and transfer learning
log_interval = 1
# CUDA
device_id = 0  # ID of GPU to use
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# parameters from command line
parser = argparse.ArgumentParser(description='Training the HDMC model')
parser.add_argument('-data_folder', type=str, default='./', help='folder for loading data and saving results')
parser.add_argument('-files', nargs='+', default=[], help='file names of different batches')
parser.add_argument('-h_thr', type=float, default=0.9, help='higher threshold for contrastive learning')
parser.add_argument('-l_thr', type=float, default=0.7, help='lower threshold for contrastive learning')
parser.add_argument('-num_epochs', type=float, default=2000, help='number of training epochs')

plt.ioff()

if __name__ == '__main__':
    # load parameters from command line
    args = parser.parse_args()
    data_folder = args.data_folder
    dataset_file_list = args.files
    cluster_similarity_file = args.data_folder+'metaneighbor.csv'
    code_save_file = args.data_folder + 'code_list.pkl'
    higher_thr = args.h_thr
    lower_thr = args.l_thr
    num_epochs = args.num_epochs
    dataset_file_list = [data_folder+f for f in dataset_file_list]

    pre_process_paras = {'take_log': True, 'standardization': True, 'scaling': True}
    nn_paras = {'code_dim': code_dim, 'batch_size': batch_size, 'num_epochs': num_epochs,
                'base_lr': base_lr, 'lr_step': lr_step,
                'momentum': momentum, 'l2_decay': l2_decay, 'gamma': gamma,
                'cuda': cuda, 'log_interval': log_interval}

    # read data
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    cluster_pairs = read_cluster_similarity(cluster_similarity_file, higher_thr, lower_thr)
    # for i in range(cluster_pairs.shape[0]):
    #     print(cluster_pairs[i])
    nn_paras['num_inputs'] = len(dataset_list[0]['gene_sym'])

    # training
    print("Training starts:")
    if len(dataset_file_list) == 2:
        model, loss_total_list, loss_reconstruct_list, loss_transfer_list, loss_adver_list\
            = training(dataset_list, cluster_pairs, nn_paras)
    else:
        model, loss_total_list, loss_reconstruct_list, loss_transfer_list, loss_adver_list \
            = multi_training(dataset_list, cluster_pairs, nn_paras)

    # save codes
    code_list = testing(model, dataset_list, nn_paras)
    codes = np.hstack((code_list[0], code_list[1]))
    if len(dataset_file_list) > 2:
        for i in range(2, len(dataset_file_list)):
            codes = np.hstack((codes, code_list[i]))
    df = pd.DataFrame(codes)
    df.to_csv(data_folder+'combined.csv')
