import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import ae as models
from mmd import mix_rbf_mmd2
import math
import time
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]
device = torch.device("cuda:0")

from imblearn.over_sampling import RandomOverSampler

imblearn_seed = 0


def training(dataset_list, cluster_pairs, nn_paras):
    """ Training an autoencoder to remove batch effects
    Args:
        dataset_list: list of datasets for batch correction
        cluster_pairs: pairs of similar clusters with weights
        nn_paras: parameters for neural network training
    Returns:
        model: trained autoencoder
        loss_total_list: list of total loss
        loss_reconstruct_list: list of reconstruction loss
        loss_transfer_list: list of transfer loss
        loss_adver_list: list of adversarial loss
    """
    # load nn parameters
    batch_size = nn_paras['batch_size']
    num_epochs = nn_paras['num_epochs']
    num_inputs = nn_paras['num_inputs']
    code_dim = nn_paras['code_dim']
    cuda = nn_paras['cuda']

    # training data for autoencoder, construct a DataLoader for each cluster
    cluster_loader_dict = {}
    for i in range(len(dataset_list)):
        gene_exp = dataset_list[i]['gene_exp'].transpose()
        cluster_labels = dataset_list[i]['cluster_labels']  # cluster labels do not overlap between datasets
        unique_labels = np.unique(cluster_labels)
        print(i, unique_labels)
        # Random oversampling based on cell cluster sizes
        gene_exp, cluster_labels = RandomOverSampler(random_state=imblearn_seed).fit_sample(gene_exp, cluster_labels)

        # construct DataLoader list
        for j in range(len(unique_labels)):
            idx = cluster_labels == unique_labels[j]
            if cuda:
                torch_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(gene_exp[idx, :]).cuda(), torch.LongTensor(cluster_labels[idx]).cuda())
            else:
                torch_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(gene_exp[idx, :]), torch.LongTensor(cluster_labels[idx]))
            data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                                      shuffle=True, drop_last=True)
            cluster_loader_dict[unique_labels[j]] = data_loader

    # create model
    if code_dim == 20:
        model = models.autoencoder_20(num_inputs=num_inputs)
    elif code_dim == 2:
        model = models.autoencoder_2(num_inputs=num_inputs)
    else:
        model = models.autoencoder_20(num_inputs=num_inputs)
    netD = models.Discriminator(num_inputs=code_dim)
    if cuda:
        model.cuda()
        netD.to(device)

    # training
    loss_total_list = []  # list of total loss
    loss_reconstruct_list = []
    loss_transfer_list = []
    loss_adver_list = []
    for epoch in range(1, num_epochs + 1):
        avg_loss, avg_reco_loss, avg_tran_loss, avg_adver_loss \
            = training_epoch(epoch, model, netD, cluster_loader_dict, cluster_pairs, nn_paras)
        # terminate early if loss is nan
        if math.isnan(avg_reco_loss) or math.isnan(avg_tran_loss):
            return [], model, [], [], []
        loss_total_list.append(avg_loss)
        loss_reconstruct_list.append(avg_reco_loss)
        loss_transfer_list.append(avg_tran_loss)
        loss_adver_list.append(avg_adver_loss)

    return model, loss_total_list, loss_reconstruct_list, loss_transfer_list, loss_adver_list


def training_epoch(epoch, model, netD, cluster_loader_dict, cluster_pairs, nn_paras):
    """ Training an epoch
        Args:
            epoch: number of the current epoch
            model: autoencoder
            netD: discriminator
            cluster_loader_dict: dict of DataLoaders indexed by clusters
            cluster_pairs: pairs of similar clusters with weights
            nn_paras: parameters for neural network training
        Returns:
            avg_total_loss: average total loss of mini-batches
            avg_reco_loss: average reconstruction loss of mini-batches
            avg_tran_loss: average transfer loss of mini-batches
            avg_adver_loss: average adversarial loss of mini-batches
        """
    log_interval = nn_paras['log_interval']
    # load nn parameters
    base_lr = nn_paras['base_lr']
    lr_step = nn_paras['lr_step']
    num_epochs = nn_paras['num_epochs']
    l2_decay = nn_paras['l2_decay']
    gamma = nn_paras['gamma']
    cuda = nn_paras['cuda']
    lamda = nn_paras['lamda']
    batch_split = nn_paras['batch_split']

    # step decay of learning rate
    learning_rate = base_lr / math.pow(2, math.floor(epoch / lr_step))
    # regularization parameter for cantrastive loss
    gamma_rate = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1
    gamma = gamma_rate * gamma

    if epoch % log_interval == 0:
        print('{:}, Epoch {}, learning rate {:.3E}, gamma {:.3E}, lamda {:.3E}'.format(
            time.asctime(time.localtime()), epoch, learning_rate, gamma, lamda))

    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': netD.parameters()}
    ], lr=learning_rate, weight_decay=l2_decay)

    # binary cross entropy loss for the discriminator
    bce = nn.BCELoss()

    model.train()

    iter_data_dict = {}
    num_iter = 0
    for cls in cluster_loader_dict:
        iter_data = iter(cluster_loader_dict[cls])
        iter_data_dict[cls] = iter_data
        num_iter = max(num_iter, len(cluster_loader_dict[cls]))

    total_loss = 0
    total_reco_loss = 0
    total_tran_loss = 0
    total_adver_loss = 0
    num_batches = 0

    for it in range(0, num_iter):
        data_dict = {}
        label_dict = {}
        code_dict = {}
        reconstruct_dict = {}
        discriminate_dict = {}
        label_list = []
        code_list = []
        real_list = []
        fake_list = []
        for cls in iter_data_dict:
            data, labels = iter_data_dict[cls].next()
            data_dict[cls] = data
            label_dict[cls] = labels
            label_list.append(labels.cpu().detach().numpy())
            if it % len(cluster_loader_dict[cls]) == 0:
                iter_data_dict[cls] = iter(cluster_loader_dict[cls])
            data_dict[cls] = Variable(data_dict[cls])
            label_dict[cls] = Variable(label_dict[cls])

        # reconstruction loss for all clusters
        loss_reconstruct = torch.FloatTensor([0]).cuda()

        for cls in data_dict:
            additional, code, reconstruct = model(data_dict[cls])
            loss_reconstruct += F.mse_loss(reconstruct, data_dict[cls])
            code_dict[cls] = code
            code_list.append(code.cpu().detach().numpy())
            reconstruct_dict[cls] = reconstruct
            reverse_code = ReverseLayerF.apply(additional, lamda)
            discriminate_dict[cls] = netD(reverse_code)
            if cls > batch_split:
                real_list.append(discriminate_dict[cls].cpu().detach().numpy())
            else:
                fake_list.append(discriminate_dict[cls].cpu().detach().numpy())

        optimizer.zero_grad()

        # contrastive loss for cluster pairs in cluster_pairs matrix
        loss_transfer = torch.FloatTensor([0])
        if cuda:
            loss_transfer = loss_transfer.cuda()
        for i in range(cluster_pairs.shape[0]):
            cls_1 = int(cluster_pairs[i, 0])
            cls_2 = int(cluster_pairs[i, 1])
            if cls_1 not in code_dict or cls_2 not in code_dict:
                continue
            mmd2_D = mix_rbf_mmd2(code_dict[cls_1], code_dict[cls_2], sigma_list)
            loss_transfer += mmd2_D * cluster_pairs[i, 2] + 0.001*(1 - mmd2_D) * (1 - cluster_pairs[i, 2])

        # adversarial loss for all samples
        real_label = 1
        fake_label = 0
        real_set = torch.from_numpy(np.array(real_list)).view(-1, 1).cuda()
        fake_set = torch.from_numpy(np.array(fake_list)).view(-1, 1).cuda()
        label = torch.full((real_set.shape[0],), real_label, dtype=real_set.dtype, device=device)
        errD_real = bce(real_set, label)
        label = torch.full((fake_set.shape[0],), fake_label, dtype=fake_set.dtype, device=device)
        errD_fake = bce(fake_set, label)
        D_loss = errD_real + errD_fake

        loss = loss_reconstruct + gamma * loss_transfer + D_loss

        loss.backward()
        optimizer.step()

        # update total loss
        num_batches += 1
        total_loss += loss.data.item()
        total_reco_loss += loss_reconstruct.data.item()
        total_tran_loss += loss_transfer.data.item()
        total_adver_loss += D_loss.data.item()

    avg_total_loss = total_loss / num_batches
    avg_reco_loss = total_reco_loss / num_batches
    avg_tran_loss = total_tran_loss / num_batches
    avg_adver_loss = total_adver_loss / num_batches

    if epoch % log_interval == 0:
        print('Avg_loss {:.3E}\t Avg_reconstruct_loss {:.3E}\t Avg_transfer_loss {:.3E}\t Avg_adver_loss {:.3E}\t'
              .format(avg_total_loss, avg_reco_loss, avg_tran_loss, avg_adver_loss))
    return avg_total_loss, avg_reco_loss, avg_tran_loss, avg_adver_loss


def testing(model, dataset_list, nn_paras):
    """ Training an epoch
    Args:
        model: autoencoder
        dataset_list: list of datasets for batch correction
        nn_paras: parameters for neural network training
    Returns:
        code_list: list pf embedded codes
    """

    # load nn parameters
    cuda = nn_paras['cuda']

    data_loader_list = []
    num_cells = []
    for dataset in dataset_list:
        torch_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(dataset['gene_exp'].transpose()), torch.LongTensor(dataset['cell_labels']))
        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=len(dataset['cell_labels']),
                                                  shuffle=False)
        data_loader_list.append(data_loader)
        num_cells.append(len(dataset["cell_labels"]))

    model.eval()

    code_list = []  # list pf embedded codes
    for i in range(len(data_loader_list)):
        idx = 0
        with torch.no_grad():
            for data, labels in data_loader_list[i]:
                if cuda:
                    data, labels = data.cuda(), labels.cuda()
                _, code_tmp, _ = model(data)
                code_tmp = code_tmp.cpu().numpy()
                if idx == 0:
                    code = np.zeros((code_tmp.shape[1], num_cells[i]))
                code[:, idx:idx + code_tmp.shape[0]] = code_tmp.T
                idx += code_tmp.shape[0]
        code_list.append(code)

    return code_list
