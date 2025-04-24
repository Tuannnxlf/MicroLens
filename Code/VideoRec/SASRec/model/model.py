import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from collections import Counter
import torch.nn.functional as F

from .user_encoders import User_Encoder_GRU4Rec, User_Encoder_SASRec, User_Encoder_NextItNet
from .Embedding2 import Embedding2

class Model(torch.nn.Module):
    def __init__(self, args, pop_prob_list, item_num, bert_model, image_net, video_net, text_content=None, pretrained_embs=None):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)

        if args.model == 'sasrec':
            self.user_encoder = User_Encoder_SASRec(args)
        elif args.model == 'gru4rec':
            self.user_encoder = User_Encoder_GRU4Rec(args)
        elif args.model == 'nextitnet':
            self.user_encoder = User_Encoder_NextItNet(args)

        self.load_item_embeddings(item_num, args.embedding_dim, pretrained_embs)

        self.criterion = nn.CrossEntropyLoss()

    def load_item_embeddings(self, item_num, embedding_dim, pretrained_embs):
        if pretrained_embs is None:
            self.id_encoder = nn.Embedding(
                num_embeddings=item_num + 1,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            xavier_normal_(self.id_encoder.weight.data)

        # use pretrained textual embedding with linear mapping as item embedding
        else:
            more_token = 0
            assert pretrained_embs.shape[0] == item_num + 1
            self.id_encoder = nn.Embedding.from_pretrained(
                torch.cat([
                    pretrained_embs,
                    torch.randn(more_token, pretrained_embs.shape[-1]).to(pretrained_embs.device)
                    ]),
                padding_idx=0
            )
            # # fix pretrained item embedding
            # self.pretrained_item_embeddings.weight.requires_grad = True

            # mlp_dims = [self.pretrained_item_embeddings.embedding_dim] + [-1]
            # mlp_dims[-1] = embedding_dim

            # # create mlp with linears and activations
            # self.item_embeddings_adapter = nn.Sequential()
            # self.item_embeddings_adapter.add_module('linear_0', nn.Linear(mlp_dims[0], mlp_dims[1]))
            # for i in range(1, len(mlp_dims) - 1):
            #     self.item_embeddings_adapter.add_module(f'activation_{i}', nn.ReLU())
            #     self.item_embeddings_adapter.add_module(f'linear_{i}', nn.Linear(mlp_dims[i], mlp_dims[i + 1]))

            # # initialize the adapter
            # for name, param in self.item_embeddings_adapter.named_parameters():
            #     if 'weight' in name:
            #         nn.init.xavier_normal_(param)
            #     elif 'bias' in name:
            #         nn.init.constant_(param, 0)
            
            # self.id_encoder = Embedding2(self.item_embeddings_adapter, self.pretrained_item_embeddings)

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def forward(self, sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args):
        self.pop_prob_list = self.pop_prob_list.to(local_rank)
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)])

        if 'modal' == args.item_tower:
            input_all_text = self.text_encoder(sample_items_text.long())
            input_all_image = self.image_encoder(sample_items_image)
            input_all_video = self.video_encoder(sample_items_video)
            input_embs = self.fusion_module(input_all_text, input_all_image, input_all_video)
        elif 'text' == args.item_tower:
            score_embs = self.text_encoder(sample_items_text.long())
        elif 'image' == args.item_tower:
            score_embs = self.image_encoder(sample_items_image)
        elif 'video' == args.item_tower:
            score_embs = self.video_encoder(sample_items_video)
        elif 'id' == args.item_tower:
            score_embs = self.id_encoder(sample_items_id)

        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)        
        if self.args.model == 'sasrec':
            prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        else:
            prec_vec = self.user_encoder(input_embs[:, :-1, :])
        prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        # logits = torch.matmul(F.normalize(prec_vec, dim=-1), F.normalize(score_embs, dim=-1).t()) # (bs * max_seq_len, bs * (max_seq_len + 1))
        # logits = logits / self.args.tau - debias_logits
        logits = torch.matmul(prec_vec, score_embs.t())
        logits = logits - debias_logits

        ###################################### MASK USELESS ITEM ######################################
        bs, seq_len = log_mask.size(0), log_mask.size(1)
        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        label = label[:, 1:].to(local_rank).view(-1)

        flatten_item_seq = sample_items_id
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        
        logits[unused_item_mask] = -1e4
        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * seq_len, -1)
        loss = self.criterion(logits[indices], label[indices])

        ###################################### CALCULATE ALIGNMENT AND UNIFORMITY ######################################
        user = prec_vec.view(-1, self.max_seq_len, self.args.embedding_dim)[:, -1, :]
        item = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, -1, :]
        align = self.alignment(user, item)
        uniform = (self.uniformity(user) + self.uniformity(item)) / 2
        
        return loss, align, uniform
