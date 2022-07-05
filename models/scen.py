import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_embedding import load_word_embeddings
from .common import MLP
import random
from itertools import product
import numpy as np
from torch import distributions as dist

from itertools import product
from simple_parsing import ArgumentParser, ConflictResolution, field

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gpu_device = 3
print('use GPU:{}'.format(gpu_device))

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights, dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i, n in enumerate(names):
            for j, m in enumerate(names):
                dict_sim[(n, m)] = similarity[i, j].item()
        return dict_sim
    return pairing_names, similarity.to('cpu')


class SCEN(nn.Module):

    def __init__(self, dset, args):
        super(SCEN, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.factor = 2

        self.scale = self.args.cosine_scale

        if dset.open_world:
            self.train_forward = self.train_forward_open
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.

            self.activated = False

            # Init feasibility-related variables
            self.attrs = dset.attrs
            self.objs = dset.objs
            self.possible_pairs = dset.pairs

            self.validation_pairs = dset.val_pairs

            self.feasibility_margin = (1 - self.seen_mask).float()
            self.epoch_max_margin = self.args.epoch_max_margin
            self.cosine_margin_factor = -args.margin

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.known_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.known_pairs:
                self.attrs_by_obj_train[o].append(a)

        else:
            self.train_forward = self.train_forward_closed
            self.dim = 512
            self.K = 1280
            self.h_dim = args.emb_dim
            self.obj_clf = nn.Linear(args.emb_dim, len(dset.objs))
            self.attr_clf = nn.Linear(args.emb_dim, len(dset.attrs))
            self.T = 0.07
            self.m = 0.999



        # Precompute training compositions
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs

        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)

        self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers,
                                  dropout=self.args.dropout,
                                  norm=self.args.norm, layers=layers)



        self.g_inv_O = MLP(int(args.emb_dim), int(args.emb_dim), relu='leaky_relu', num_layers=args.nlayers,
                              dropout=self.args.dropout,
                              norm=self.args.norm, layers=layers).to(device)  # object

        layers = [1024, 1200]
        self.g_inv_A = MLP(int(args.emb_dim), int(args.emb_dim), relu='leaky_relu', num_layers=args.nlayers,
                              dropout=self.args.dropout,
                              norm=self.args.norm, layers=layers).to(device)  # state

        self.image_decoder = MLP(int(args.emb_dim) * 2, dset.feat_dim, relu=args.relu,
                                          num_layers=args.nlayers,
                                          dropout=self.args.dropout,
                                          norm=self.args.norm, layers=layers)


        # Fixed
        self.composition = args.composition

        input_dim = args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)

        # init with word embeddings
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # static inputs
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        # Composition MLP
        self.projection = nn.Linear(input_dim * 2, args.emb_dim)
        self.projection_1 = nn.Linear(input_dim * 2, args.emb_dim)

    def freeze_representations(self):
        print('Freezing representations')
        for param in self.image_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False
        for param in self.projection_1.parameters():
            param.requires_grad = False
        for param in self.image_decoder.parameters():
            param.requires_grad = False

    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)

        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        output = F.normalize(output, dim=1)
        return output

    def compose_1(self, obj_feats, att_feats):

        inputs = torch.cat([obj_feats, att_feats], 1)
        output = self.projection_1(inputs)
        output = F.normalize(output, dim=1)
        return output

    def WeightedL1(self, pred, gt):
        # embedding cycle-consistency loss
        wt = (pred - gt).pow(2)
        wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
        loss = wt * (pred - gt).abs()
        return loss.sum() / loss.size(0)

    def compute_feasibility(self):
        obj_embeddings = self.obj_embedder(torch.arange(len(self.objs)).long().to('cuda'))
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings,
                                                      return_dict=True)
        attr_embeddings = self.attr_embedder(torch.arange(len(self.attrs)).long().to('cuda'))
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings,
                                                       return_dict=True)

        feasibility_scores = self.seen_mask.clone().float()
        for a in self.attrs:
            for o in self.objs:
                if (a, o) not in self.known_pairs:
                    idx = self.dset.all_pair2idx[(a, o)]
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    feasibility_scores[idx] = score

        self.feasibility_scores = feasibility_scores

        return feasibility_scores * (1 - self.seen_mask.float())

    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.
        for o in self.objs:
            if o != obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[(obj, o)]
                if temp_score > score:
                    score = temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[(attr, a)]
                if temp_score > score:
                    score = temp_score
        return score

    def update_feasibility(self, epoch):
        self.activated = True
        feasibility_scores = self.compute_feasibility()
        self.feasibility_margin = min(1., epoch / self.epoch_max_margin) * \
                                  (self.cosine_margin_factor * feasibility_scores.float().to(device))


    def val_forward(self, x):
        img = x[0]
        img = self.image_embedder(img)
        obj_feats = self.g_inv_O(img)
        att_feats = self.g_inv_A(img)
        img_feats = self.compose_1(obj_feats, att_feats)

        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)  # Evaluate all pairs
        score = torch.matmul(img_feats_normed, pair_embeds)

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]


        return None, scores
        # return None, scores, scores_obj, scores_att

    # def val_forward_with_threshold(self, x, th=0.):
    #     img = x[0]
    #     img_feats = self.image_embedder(img)
    #     img_feats_normed = F.normalize(img_feats, dim=1)
    #     pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)  # Evaluate all pairs
    #     score = torch.matmul(img_feats_normed, pair_embeds)
    #
    #     # Note: Pairs are already aligned here
    #     mask = (self.feasibility_scores >= th).float()
    #     score = score * mask + (1. - mask) * (-1.)
    #
    #     scores = {}
    #     for itr, pair in enumerate(self.dset.pairs):
    #         scores[pair] = score[:, self.dset.all_pair2idx[pair]]
    #
    #     return None, scores
    #
    # def train_forward_open(self, x):
    #     img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
    #     img_feats = self.image_embedder(img)
    #
    #     pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)
    #     img_feats_normed = F.normalize(img_feats, dim=1)
    #
    #     pair_pred = torch.matmul(img_feats_normed, pair_embed)
    #
    #     if self.activated:
    #         pair_pred += (1 - self.seen_mask) * self.feasibility_margin
    #         loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)
    #     else:
    #         pair_pred = pair_pred * self.seen_mask + (1 - self.seen_mask) * (-10)
    #         loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)
    #
    #     return loss_cos.mean(), None

    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3] #img: extracted features by resnet18 img_trans: transformed_img
        neg_attrs, neg_objs = x[4][:,0], x[5][:,0] #todo:do a less hacky version
        img_pos_obj, img_pos_att = x[8], x[9]

        img_resize = self.image_embedder(img)
        img_pos_obj = self.image_embedder(img_pos_obj)
        img_pos_att = self.image_embedder(img_pos_att)

        h_A, h_O = self.attr_embedder(self.train_attrs).permute(1, 0), self.obj_embedder(self.train_objs).permute(1, 0)
        obj_feats = self.g_inv_O(img_resize) # size: 512 to 300
        att_feats = self.g_inv_A(img_resize)
        obj_feats_normed = F.normalize(obj_feats, dim=1)
        att_feats_normed = F.normalize(att_feats, dim=1)
        index_1 = random.choice(range(img_pos_obj.shape[1]))
        pos_obj_sample = img_pos_obj[:, index_1, :]
        obj_pos_feats = self.g_inv_O(pos_obj_sample)
        obj_pos_feats_normed = F.normalize(obj_pos_feats, dim=1)
        obj_pred = torch.matmul(obj_feats_normed, h_O)

        loss_con_obj = 0
        for i in range(len(img_pos_att[0])):
            neg_obj_sample = img_pos_att[:, i, :]
            obj_neg_feats = self.g_inv_O(neg_obj_sample)
            obj_neg_feats_normed = F.normalize(obj_neg_feats, dim=1)
            # obj_neg_pred = torch.matmul(obj_neg_feats_normed, h_O)
            loss_con_obj += F.triplet_margin_loss(obj_feats_normed, obj_pos_feats_normed, obj_neg_feats_normed, margin=self.args.margin)
        loss_con_obj /= len(img_pos_att)

        index_1 = random.choice(range(img_pos_att.shape[1]))
        pos_att_sample = img_pos_att[:, index_1, :]
        att_pos_feats = self.g_inv_A(pos_att_sample)
        att_pos_feats_normed = F.normalize(att_pos_feats, dim=1)
        attr_pred = torch.matmul(att_feats_normed, h_A)

        loss_con_att = 0
        for i in range(len(img_pos_obj[0])):
            neg_att_sample = img_pos_obj[:, i, :]
            att_neg_feats = self.g_inv_A(neg_att_sample)
            att_neg_feats_normed = F.normalize(att_neg_feats, dim=1)
            # att_neg_pred = torch.matmul(att_neg_feats_normed, h_A)
            loss_con_att += F.triplet_margin_loss(att_feats_normed, att_pos_feats_normed, att_neg_feats_normed, margin=self.args.margin)
        loss_con_att /= len(img_pos_obj)

        loss_aux = 0.4 * F.cross_entropy(self.scale * attr_pred, attrs).mean() + 0.6 * F.cross_entropy(self.scale * obj_pred, objs).mean()
        loss_con = loss_con_obj + loss_con_att

        img_feats = self.compose_1(obj_feats, att_feats)

        pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_pred = torch.matmul(img_feats_normed, pair_embed)

        loss_cls = F.cross_entropy(self.scale * pair_pred, pairs).mean()

        loss = loss_cls + self.args.lambda_aux * loss_aux + self.args.lambda_con * loss_con
        # print("loss_cls:", loss_cls)
        # print("loss_aux:", loss_aux)
        # print("loss_con:", loss_con)
        # print("loss_con_att:", loss_con_att)
        # print("loss_con_obj:", loss_con_obj)

        return  loss, None

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr