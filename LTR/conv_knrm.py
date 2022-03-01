import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as f
from torch.autograd import Variable
import torch.nn.functional as F


class CONVKNRM(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(CONVKNRM, self).__init__()

        self.wv=args.wv
        self.index_to_word=args.index_to_word

        self.input_dim=args.emsize
        self.device=args.device

        self.nbins = args.nbins

        self.dense_f = nn.Linear(self.nbins * 9, 1, 1)
        self.tanh = nn.Tanh()


        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, self.input_dim)),
            nn.ReLU()
        )

        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, self.input_dim)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, self.input_dim)),
            nn.ReLU()
        )

        tensor_mu = torch.FloatTensor(args.mu).to(self.device)
        tensor_sigma = torch.FloatTensor(args.sigma).to(self.device)

        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, self.nbins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, self.nbins)

    def get_intersect_matrix(self, q_embed, d_embed):
        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum


    def to_embedding(self,input):
        shape_input = list(input.shape)

        em = input.view(-1)
        list_of_embeddings = []
        for key in em:
            list_of_embeddings += self.wv[self.index_to_word[key]]
        list_of_embeddings = torch.Tensor(list_of_embeddings)
        embeds = list_of_embeddings.view(shape_input[0], shape_input[1],
                                         self.input_dim).to(self.device)

        # embeds = self.word_embeddings(input)
        ss = embeds.shape
        # embeds = embeds.view(ss[0],-1, ss[3], self.input_dim)

        #emb = torch.squeeze(embeds)  # batch_size * 1 * seq_len * emb_dim
        return embeds


    def forward(self, batch_queries, batch_docs,batch_semantic):


        qlen = batch_queries.shape[1]
        num_docs, dlen = batch_docs.shape[0], batch_docs.shape[1]

        batch_size=1


        emb_query = self.to_embedding(batch_queries)
        emb_desc = self.to_embedding(batch_docs)

        desc_att_shape = emb_desc.shape
        query_shape = emb_query.shape

        qwu_embed = torch.transpose(
            torch.squeeze(self.conv_uni(emb_query.view(emb_query.size()[0], 1, -1, self.input_dim))), 1,
            2) + 0.000000001

        qwb_embed = torch.transpose(
            torch.squeeze(self.conv_bi(emb_query.view(emb_query.size()[0], 1, -1, self.input_dim))), 1,
            2) + 0.000000001
        qwt_embed = torch.transpose(
            torch.squeeze(self.conv_tri(emb_query.view(emb_query.size()[0], 1, -1, self.input_dim))), 1,
            2) + 0.000000001


        dwu_embed = torch.squeeze(
            self.conv_uni(emb_desc.view(emb_desc.size()[0], 1, -1, self.input_dim))) + 0.000000001
        dwb_embed = torch.squeeze(
            self.conv_bi(emb_desc.view(emb_desc.size()[0], 1, -1, self.input_dim))) + 0.000000001
        dwt_embed = torch.squeeze(
            self.conv_tri(emb_desc.view(emb_desc.size()[0], 1, -1, self.input_dim))) + 0.000000001

        qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        qwb_embed_norm = F.normalize(qwb_embed, p=2, dim=2, eps=1e-10)
        qwt_embed_norm = F.normalize(qwt_embed, p=2, dim=2, eps=1e-10)
        dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)
        dwb_embed_norm = F.normalize(dwb_embed, p=2, dim=1, eps=1e-10)
        dwt_embed_norm = F.normalize(dwt_embed, p=2, dim=1, eps=1e-10)

        log_pooling_sum_wwuu = self.get_intersect_matrix(qwu_embed_norm, dwu_embed_norm)
        log_pooling_sum_wwut = self.get_intersect_matrix(qwu_embed_norm, dwt_embed_norm)
        log_pooling_sum_wwub = self.get_intersect_matrix(qwu_embed_norm, dwb_embed_norm)
        log_pooling_sum_wwbu = self.get_intersect_matrix(qwb_embed_norm, dwu_embed_norm)
        log_pooling_sum_wwtu = self.get_intersect_matrix(qwt_embed_norm, dwu_embed_norm)

        log_pooling_sum_wwbb = self.get_intersect_matrix(qwb_embed_norm, dwb_embed_norm)
        log_pooling_sum_wwbt = self.get_intersect_matrix(qwb_embed_norm, dwt_embed_norm)
        log_pooling_sum_wwtb = self.get_intersect_matrix(qwt_embed_norm, dwb_embed_norm)
        log_pooling_sum_wwtt = self.get_intersect_matrix(qwt_embed_norm, dwt_embed_norm)

        log_pooling_sum = torch.cat(
            [log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu,
             log_pooling_sum_wwtu,log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)

        output = torch.squeeze(F.tanh(self.dense_f(log_pooling_sum)), 1)



        return output

