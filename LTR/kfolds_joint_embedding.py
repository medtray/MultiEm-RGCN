import torch.nn.functional as F
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cuda'
from joint_model import JointModel
from data_reader_jm import DataAndQueryJM
import os
import numpy as np
import torch.nn.functional as F
import pandas as pd
import subprocess
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import random
import argparse
import sys
cwd = os.getcwd()
from pathlib import Path
path = Path(cwd)
sys.path.append(os.path.join(path.parent.absolute()))

parser = argparse.ArgumentParser(description='Train MultiEm-RGCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--emsize', type=int, default=100)
parser.add_argument('--max_query_len', type=int, default=150)
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--filters_1d', type=int, nargs='+', default=[100], help='Decrease learning rate at these epochs.')
parser.add_argument('--kernel_size_1d', type=int, nargs='+', default=[5], help='Decrease learning rate at these epochs.')
parser.add_argument('--maxpool_size_1d', type=int, nargs='+', default=[2], help='Decrease learning rate at these epochs.')

parser.add_argument('--nbins', type=int, default=5)

parser.add_argument("--n-hidden", type=int, default=100,
                        help="number of hidden units")

parser.add_argument("--n-bases", type=int, default=10,
                    help="number of weight blocks for each relation")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")

parser.add_argument("--dataset", type=str, default='wikiTables',
                    help="dataset to use")
parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")



parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

args = parser.parse_args()
print(torch.cuda.current_device())
torch.cuda.set_device(args.device)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(args.device))
print(torch.cuda.is_available())
args.device='cuda:'+str(args.device)
args.use_cuda=True

# args.use_cuda=False
# args.device='cpu'

out_str = str(args)
print(out_str)

loss_function=nn.MSELoss()

import os
from pathlib import Path


def kernal_mus(n_kernels):
    """
    get the mu for each gaussian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each gaussian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

args.mu = kernal_mus(args.nbins)
args.sigma = kernel_sigmas(args.nbins)

cwd = os.getcwd()

parent_path=Path(cwd).parent
args.parent_path=parent_path
data_folder=os.path.join(parent_path,args.dataset)
text_file = open(os.path.join(data_folder,"qrels.txt"), "r")
lines = text_file.readlines()

queries_id_qrels = []
list_lines_qrels = []

for line in lines:
    # print(line)
    line = line[0:len(line) - 1]
    aa = line.split('\t')
    queries_id_qrels += [aa[0]]
    list_lines_qrels.append(aa)

def load_checkpoint(model, optimizer, losslogger, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger

def load_checkpoint_for_eval(model, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model

def read_file_for_nfcg(file):
    text_file = open(file, "r")
    lines = text_file.readlines()

    queries_id = []
    list_lines = []

    for line in lines:
        # print(line)
        line = line[0:len(line) - 1]
        aa = line.split('\t')
        queries_id += [aa[0]]
        list_lines.append(aa)
    inter = np.array(list_lines)

    return inter

def test_output(test_iter, model):

    #model = load_checkpoint_for_eval(model, save_path)
    # move the model to GPU if has one
    model=model.to(args.device)

    # need this for dropout
    model.eval()

    epoch_loss = 0
    num_batches = len(test_iter)
    all_outputs = []
    all_labels=[]
    for batch_desc_w,batch_att_w,batch_query_w,batch_desc_wn,batch_att_wn,batch_query_wn,batch_desc_QTE,batch_query_QTE,\
        labels,batch_semantic in test_iter:
        batch_desc_w, batch_att_w, batch_query_w, labels,batch_semantic = batch_desc_w.to(args.device), batch_att_w.to(args.device),\
                            batch_query_w.to(args.device), labels.to(args.device), batch_semantic.to(args.device)

        batch_desc_wn, batch_att_wn, batch_query_wn = batch_desc_wn.to(args.device), batch_att_wn.to(args.device), \
                                                      batch_query_wn.to(args.device)

        batch_desc_QTE, batch_query_QTE = batch_desc_QTE.to(args.device),batch_query_QTE.to(args.device)

        batch_query_w = torch.squeeze(batch_query_w)
        batch_desc_w = torch.squeeze(batch_desc_w)
        batch_att_w = torch.squeeze(batch_att_w)

        batch_query_wn = torch.squeeze(batch_query_wn)
        batch_desc_wn = torch.squeeze(batch_desc_wn)
        batch_att_wn = torch.squeeze(batch_att_wn)

        batch_query_QTE = torch.squeeze(batch_query_QTE)
        batch_desc_QTE = torch.squeeze(batch_desc_QTE)


        batch_desc_w = torch.cat([batch_desc_w, batch_att_w], 1)
        batch_desc_wn = torch.cat([batch_desc_wn, batch_att_wn], 1)

        outputs = model(batch_query_w,batch_desc_w,batch_query_wn,batch_desc_wn,batch_query_QTE, batch_desc_QTE,batch_semantic).to(args.device)
        # print(outputs)
        # labels=torch.FloatTensor(labels)
        #labels = labels / 2

        loss = loss_function(outputs, labels.float())
        #loss = listnet_loss(labels.float(), outputs)
        epoch_loss += loss.item()

        all_outputs += outputs.tolist()
        all_labels += labels.tolist()

    losslogger = epoch_loss / num_batches

    #print(f'Testing loss = {losslogger}')

    return all_outputs,losslogger,all_labels


def calculate_metrics(inter, output_file,all_outputs,ndcg_file):
    inter2 = []

    for jj, item in enumerate(inter):
        item_inter = [i for i in item]
        item_inter[4] = str(all_outputs[jj])

        inter2.append(item_inter)

    inter3 = np.array(inter2)

    np.savetxt(output_file, inter3, fmt="%s")

    #batcmd = "./trec_eval -m ndcg_cut.5 "+ndcg_file+" " + output_file
    batcmd = "./trec_eval -m map " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    map = float(res[2])

    batcmd = "./trec_eval -m recip_rank " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    mrr = float(res[2])

    batcmd = "./trec_eval -m ndcg_cut.5 " + ndcg_file + " " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    ndcg = float(res[2])

    return ndcg,map,mrr




def calculate_ndcg(inter, output_file,all_outputs,ndcg_file):
    inter2 = []

    for jj, item in enumerate(inter):
        item_inter = [i for i in item]
        item_inter[4] = str(all_outputs[jj])

        inter2.append(item_inter)

    inter3 = np.array(inter2)

    np.savetxt(output_file, inter3, fmt="%s")

    batcmd = "./trec_eval -m ndcg_cut.5 "+ndcg_file+" " + output_file
    result = subprocess.check_output(batcmd, shell=True, encoding='cp437')
    res = result.split('\t')
    ndcg = float(res[2])

    return ndcg

def qrel_for_data(data,list_lines_qrels,output_file):
    #list_lines_qrels=np.array(list_lines_qrels)
    df = pd.DataFrame(list_lines_qrels)
    qrel_inter=[]
    for i in range(len(data)):
        row=data[i]
        ii=df[((df[0] == row[0]) & (df[2] == row[2]))]
        qrel_inter+=ii.values.tolist()

    qrel_inter=np.array(qrel_inter)

    np.savetxt(output_file, qrel_inter, fmt="%s",delimiter='\t')

def listnet_loss(y_i, z_i):
    """
    y_i: (n_i, 1)
    z_i: (n_i, 1)
    """

    P_y_i = F.softmax(y_i, dim=0)
    P_z_i = F.softmax(z_i, dim=0)
    return - torch.sum(y_i * torch.log(P_z_i))

loss_function=nn.MSELoss()


batch_size=50

kfold = KFold(5, True, None)
data=read_file_for_nfcg(os.path.join(data_folder,"all.txt"))

#all_ind=np.arange(len(data))
#random.shuffle(all_ind)
#data=data[all_ind]

NUM_EPOCH=5

start_epoch=0
d_dropout = 0.2

final_results=[]
final_results_map=[]
final_results_mrr=[]

for _ in range(3):

    all_test_max_ndcg = []
    all_test_max_map = []
    all_test_max_mrr = []

    split_id = 0

    for train, test in kfold.split(data):
        ndcg_train = []
        ndcg_test = []

        split_id+=1

        output_qrels_train='qrels_train'+str(split_id)+'.txt'
        qrel_for_data(data[train], list_lines_qrels, output_qrels_train)
        output_qrels_test = 'qrels_test'+str(split_id)+'.txt'
        qrel_for_data(data[test], list_lines_qrels, output_qrels_test)

        train_file_name = './train1_'+str(split_id)+'.txt'
        np.savetxt(train_file_name, data[train], fmt="%s",delimiter='\t')
        test_file_name = './test1_'+str(split_id)+'.txt'
        np.savetxt(test_file_name, data[test], fmt="%s",delimiter='\t')

        output_train_ndcg = './train1_ndcg_'+str(split_id)+'.txt'
        output_test_ndcg = './test1_ndcg_'+str(split_id)+'.txt'

        train_dataset = DataAndQueryJM(train_file_name,None,None,None,output_train_ndcg,args)
        print(len(train_dataset))
        train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

        args.index_to_word=train_dataset.index_to_word
        args.wv=train_dataset.wv

        model = JointModel(args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-8)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-8)
        losslogger=np.inf
        save_path = './model2.pt'

        #model, optimizer, start_epoch, losslogger=load_checkpoint(model, optimizer, losslogger, save_path)


        test_dataset = DataAndQueryJM(test_file_name, train_dataset.wv, train_dataset.word_to_index,
                                        train_dataset.index_to_word,output_test_ndcg,args)
        print(len(test_dataset))
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        inter_train = read_file_for_nfcg(output_train_ndcg)
        inter_test = read_file_for_nfcg(output_test_ndcg)

        all_att_w = train_dataset.all_att_w
        all_desc_w = train_dataset.all_desc_w
        all_query_w = train_dataset.all_query_w

        all_att_wn = train_dataset.all_att_wn
        all_desc_wn = train_dataset.all_desc_wn
        all_query_wn = train_dataset.all_query_wn

        all_desc_QTE = train_dataset.all_desc_QTE
        all_query_QTE = train_dataset.all_query_QTE

        all_semantic = train_dataset.all_semantic

        all_query_labels = train_dataset.all_query_labels
        all_labels = np.array(train_dataset.labels)


        dict_label_pos = {}

        for l in range(1, 61):
            dict_label_pos[l] = [i for i, num in enumerate(all_query_labels) if num == l]

        loss_train=[]
        loss_test=[]
        max_test_ndcg=0
        max_test_map = 0
        max_test_mrr = 0

        for epoch in range(start_epoch, NUM_EPOCH + start_epoch):
            model.train()
            epoch_loss = 0
            # num_batches = len(train_iter)
            num_batches = 60
            all_outputs = []

            for l in range(1, 61):

                if len(dict_label_pos[l]) > 0:
                    batch_desc_w = all_desc_w[dict_label_pos[l]].to(args.device)
                    batch_att_w = all_att_w[dict_label_pos[l]].to(args.device)
                    batch_query_w = all_query_w[dict_label_pos[l]].to(args.device)

                    batch_desc_wn = all_desc_wn[dict_label_pos[l]].to(args.device)
                    batch_att_wn = all_att_wn[dict_label_pos[l]].to(args.device)
                    batch_query_wn = all_query_wn[dict_label_pos[l]].to(args.device)

                    batch_desc_QTE = all_desc_QTE[dict_label_pos[l]].to(args.device)
                    batch_query_QTE = all_query_QTE[dict_label_pos[l]].to(args.device)

                    batch_semantic = all_semantic[dict_label_pos[l]].to(args.device)


                    labels = torch.tensor(all_labels[dict_label_pos[l]])
                    labels = labels.to(args.device)

                    batch_query_w=torch.squeeze(batch_query_w)
                    batch_desc_w = torch.squeeze(batch_desc_w)
                    batch_att_w = torch.squeeze(batch_att_w)

                    batch_query_wn = torch.squeeze(batch_query_wn)
                    batch_desc_wn = torch.squeeze(batch_desc_wn)
                    batch_att_wn = torch.squeeze(batch_att_wn)

                    batch_query_QTE = torch.squeeze(batch_query_QTE)
                    batch_desc_QTE = torch.squeeze(batch_desc_QTE)


                    batch_desc_w=torch.cat([batch_desc_w,batch_att_w],1)
                    batch_desc_wn = torch.cat([batch_desc_wn, batch_att_wn], 1)


                    outputs = model(batch_query_w,batch_desc_w,batch_query_wn,batch_desc_wn,batch_query_QTE,batch_desc_QTE
                                    ,batch_semantic).to(args.device)
                    # labels=torch.FloatTensor(labels)

                    all_outputs += outputs.tolist()

                    # labels=labels/2

                    # loss = loss_function(outputs, labels.float()).to(device)
                    loss = listnet_loss(labels.float(), outputs).to(args.device)
                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


            losslogger = epoch_loss / num_batches

            #print(f'Training epoch = {epoch + 1}, epoch loss = {losslogger}')

            # train_ndcg = calculate_ndcg(inter_train, 'scores.txt', all_outputs, 'qrels.txt')
            train_ndcg = calculate_ndcg(inter_train, 'scores.txt', all_outputs, output_qrels_train)
            ndcg_train.append(train_ndcg)
            #print(ndcg_train)

            outputs_test, testing_loss, _ = test_output(test_iter, model)
            # test_ndcg = calculate_ndcg(inter_test, 'scores.txt', outputs_test, 'qrels.txt')
            test_ndcg2 = calculate_ndcg(inter_test, 'scores.txt', outputs_test, output_qrels_test)
            test_ndcg,test_map,test_mrr = calculate_metrics(inter_test, 'scores.txt', outputs_test, output_qrels_test)
            if test_ndcg>max_test_ndcg:
                max_test_ndcg=test_ndcg

            # ndcg_test.append(test_ndcg)
            # print(max_test_ndcg)
            #
            # if test_map>max_test_map:
            #     max_test_map=test_map
            #
            # if test_mrr>max_test_mrr:
            #     max_test_mrr=test_mrr

            loss_train.append(losslogger)
            loss_test.append(testing_loss)
            #print(model.SRscore.weight)

        all_test_max_ndcg.append(test_ndcg)
        print(all_test_max_ndcg)

        all_test_max_map.append(test_map)
        #print(all_test_max_map)

        all_test_max_mrr.append(test_mrr)
        #print(all_test_max_mrr)

    final_results+=all_test_max_ndcg
    final_results_map += all_test_max_map
    final_results_mrr += all_test_max_mrr

print('final results \n')

print(final_results)
print(len(final_results))
print('mean ndcg={}'.format(np.mean(final_results)))
print('std ndcg={}'.format(np.std(final_results)))

print(final_results_map)
print(len(final_results_map))
print('mean map={}'.format(np.mean(final_results_map)))
print('std map={}'.format(np.std(final_results_map)))

print(final_results_mrr)
print(len(final_results_mrr))
print('mean mrr={}'.format(np.mean(final_results_mrr)))
print('std mrr={}'.format(np.std(final_results_mrr)))





