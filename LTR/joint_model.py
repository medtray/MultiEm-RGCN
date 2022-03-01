import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ARCI(nn.Module):
    def __init__(self, args):
        """"Constructor of the class."""
        super(ARCI, self).__init__()

        self.wv = args.wv
        self.index_to_word = args.index_to_word

        self.input_dim = args.emsize
        self.device = args.device
        #self.emb_drop = nn.Dropout(p=args.dropout_emb)

        num_conv1d_layers = len(args.filters_1d)
        assert num_conv1d_layers == len(args.kernel_size_1d)
        assert num_conv1d_layers == len(args.maxpool_size_1d)

        query_feats = args.max_query_len

        query_conv1d_layers = []
        doc_conv1d_layers = []
        for i in range(num_conv1d_layers):
            inpsize = args.emsize if i == 0 else args.filters_1d[i - 1]
            pad = args.kernel_size_1d[i] // 2
            layer = nn.Sequential(
                nn.Conv1d(inpsize, args.filters_1d[i], args.kernel_size_1d[i],
                          padding=pad),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(args.maxpool_size_1d[i])
            )
            query_conv1d_layers.append(layer)

            query_feats = query_feats // args.maxpool_size_1d[i]
            assert query_feats != 0

        self.query_conv1d_layers = nn.ModuleList(query_conv1d_layers)

        #inpsize = (args.filters_1d[-1] * query_feats) + self.input_dim
        inpsize = (args.filters_1d[-1] * query_feats)
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.input_dim, 1),
        #     #nn.Linear(inpsize // 2, 1)
        # )

        self.mlpQ=nn.Linear(inpsize,self.input_dim)


    def to_embedding(self,input):
        shape_input = list(input.shape)

        em = input.view(-1)
        list_of_embeddings = []
        for key in em:
            list_of_embeddings += self.wv[self.index_to_word[key]]
        list_of_embeddings = torch.Tensor(list_of_embeddings)
        embeds = list_of_embeddings.view(shape_input[0], shape_input[1],
                                         self.input_dim).to(self.device)

        return embeds

    def to_embedding_doc(self,input):
        shape_input = list(input.shape)

        em = input.view(-1)
        list_of_embeddings = []
        for key in em:
            list_of_embeddings += self.wv[self.index_to_word[key]]
        list_of_embeddings = torch.Tensor(list_of_embeddings)
        embeds = list_of_embeddings.view(shape_input[0],self.input_dim).to(self.device)

        return embeds

    def forward(self, batch_queries, batch_docs):

        #batch_docs = batch_docs.unsqueeze(0)
        batch_queries = batch_queries[0:1]
        num_docs = batch_docs.shape[0]

        batch_size = 1

        embedded_queries = self.to_embedding(batch_queries)


        embedded_queries = embedded_queries[0:1]

        inp_rep = embedded_queries.transpose(1, 2)
        for layer in self.query_conv1d_layers:
            inp_rep = layer(inp_rep)
        # batch_size x ?
        conv_queries = inp_rep.flatten(1)

        # batch_size x num_rel_docs x ?
        conv_queries = conv_queries.unsqueeze(1).expand(
            batch_size, num_docs, conv_queries.size(1))
        # batch_size * num_rel_docs x ?
        conv_queries = conv_queries.contiguous().view(batch_size * num_docs, -1)

        # embedded_queries_mean=torch.mean(embedded_queries,dim=1)
        # # batch_size x num_rel_docs x ?
        # conv_queries = embedded_queries_mean.unsqueeze(1).expand(
        #     batch_size, num_docs, embedded_queries_mean.size(1))
        # batch_size * num_rel_docs x ?
        conv_queries = conv_queries.contiguous().view(batch_size * num_docs, -1)

        conv_queries=self.mlpQ(conv_queries)

        embedded_docs = self.to_embedding_doc(batch_docs)

        #com_rep = torch.cat((conv_queries, embedded_docs), 1)
        com_rep= torch.mul(conv_queries,embedded_docs)
        return com_rep



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

        #self.dense_f = nn.Linear(self.nbins * 9, 1, 1)
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

        return embeds


    def forward(self, batch_queries, batch_docs,batch_semantic):


        emb_query = self.to_embedding(batch_queries)
        emb_desc = self.to_embedding(batch_docs)

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

        #output = torch.squeeze(F.tanh(self.dense_f(log_pooling_sum)), 1)



        return log_pooling_sum


class JointModel(nn.Module):
    def __init__(self, args):
        """"Constructor of the class."""
        super(JointModel, self).__init__()

        self.output = nn.Linear(75, 1, 1)
        self.convknrm = CONVKNRM(args).to(args.device)
        self.arci = ARCI(args).to(args.device)
        self.args=args

        self.fc1 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(45, 20)

    def forward(self, batch_queries_w, batch_docs_w,batch_queries_wn,batch_docs_wn,batch_queries_t,batch_docs_t, batch_semantic):
        outputs_w = self.convknrm(batch_queries_w, batch_docs_w, batch_semantic).to(self.args.device)
        outputs_wn = self.convknrm(batch_queries_wn, batch_docs_wn, batch_semantic).to(self.args.device)
        outputs_t = self.arci(batch_queries_t, batch_docs_t).to(self.args.device)
        outputs_t = self.fc1(outputs_t)
        outputs_wn = self.fc2(outputs_wn)
        feat=torch.cat([outputs_w,outputs_wn,outputs_t],dim=1)

        scores=self.output(feat)
        scores=torch.squeeze(scores,dim=1)

        return scores

