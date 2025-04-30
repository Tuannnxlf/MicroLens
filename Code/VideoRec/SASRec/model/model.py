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

        self.id_encoder = nn.Embedding(
                num_embeddings=item_num + 1,
                embedding_dim=args.embedding_dim,
                padding_idx=0
            )
        xavier_normal_(self.id_encoder.weight.data)
        
        # Embedding.from_pretrained
        more_token = 0
        assert pretrained_embs.shape[0] == item_num + 1
        self.pretrained_item_embeddings = nn.Embedding.from_pretrained(
            torch.cat([
                pretrained_embs,
                torch.randn(more_token, pretrained_embs.shape[-1]).to(pretrained_embs.device)
                ]),
            padding_idx=0
        )
        # fix pretrained item embedding
        self.pretrained_item_embeddings.weight.requires_grad = False
        self.pretrained_item_embeddings.weight[-more_token:].requires_grad = True

        self.pathway = nn.Linear(self.pretrained_item_embeddings.embedding_dim, args.embedding_dim)
        self.gating = nn.Sequential(
            nn.Linear(args.embedding_dim * 2, args.embedding_dim),
            nn.Sigmoid()
        )

        self.criterion = nn.CrossEntropyLoss()

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def id_pretrained_fusion(self, sample_items_id):
        id_embs = self.id_encoder(sample_items_id)
        pretrained_embs = self.pretrained_item_embeddings(sample_items_id)
        pretrained_embs = torch.relu(self.pathway(pretrained_embs))
        fused = torch.cat((id_embs, pretrained_embs), dim=1)
        gate = self.gating(fused)
        score_embs = gate * id_embs + (1 - gate) * pretrained_embs
        return score_embs, gate

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
            score_embs, gate = self.id_pretrained_fusion(sample_items_id)

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
        
        return loss, align, uniform, gate
