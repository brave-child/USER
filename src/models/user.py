import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from collections import defaultdict
import math
from scipy.sparse import lil_matrix
import random
import json

class USER(GeneralRecommender):
    def __init__(self, config, dataset, local_time):
        super(USER, self).__init__(config, dataset)
        self.sparse = True
        self.sc_loss = config['sc_loss']
        self.mp_loss = config['mp_loss']
        self.ia_loss = config['ia_loss']
        self.reg_weight_1 = config['reg_weight_1']
        self.reg_weight_2 = config['reg_weight_2']
        self.sc_temp = config['sc_temp']
        self.mp_temp = config['mp_temp']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.raw_item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.raw_item_embedding.weight)
        
        self.latent_v_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.latent_v_user.weight)
        
        self.latent_t_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.latent_t_user.weight)

        self.dataset_path = os.path.abspath(os.getcwd()+config['data_path'] + config['dataset'])
        self.data_name = config['dataset']

        image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        if self.v_feat is not None:
            self.v_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.v_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.t_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.t_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        self.inter = self.co_perception(self.image_original_adj, self.text_original_adj)
        self.co_ii_adj = self.add_edge(self.inter)
        self.co_ui_adj = self.get_adj_mat(self.co_ii_adj.tolil())

        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.co_ui_adj = self.sparse_mx_to_torch_sparse_tensor(self.co_ui_adj).float().to(self.device)
        
        
        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.v_mlp = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )
        
        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.t_mlp = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )

        self.fgpe = FGPEncoder(self.embedding_dim)

        visual_tokens = torch.load("codebook/visual_codebook.pth")
        textual_tokens = torch.load("codebook/textual_codebook.pth")
        self.visual_token_embedding = torch.nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False).cuda()
        self.text_token_embedding = torch.nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False).cuda()
        
        self.ma_decoder = MA_Decoder()


    def co_perception(self, image_adj, text_adj):
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim = []
            txt_sim = []
            for i in range(0,len(image_adj._indices()[0])):
                img_id = image_adj._indices()[0][i]
                txt_id = text_adj._indices()[0][i]
                assert img_id == txt_id
                id = img_id.item()
                img_sim.append(image_adj._indices()[1][j].item())
                txt_sim.append(text_adj._indices()[1][j].item())
                
                if len(img_sim)==10 and len(txt_sim)==10:
                    it_inter = list(set(img_sim) & set(txt_sim))
                    inter[id] = [v for v in it_inter if v != id]
                    img_sim = []
                    txt_sim = []
                
                j += 1
            
            with open(inter_file, "w") as f:
                json.dump(inter, f)
        
        return inter

    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)
        
        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1]*len(sim_rows)

        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items,self.n_items), dtype=np.int)
        return item_adj
    
    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj
        
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def ui_gcn(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
        return all_embeddings

    def ii_gcn(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def latent_preference_extract(self):
        v_item_embeds = torch.multiply(self.raw_item_embedding.weight, self.v_mlp(self.v_embedding.weight))
        t_item_embeds = torch.multiply(self.raw_item_embedding.weight, self.t_mlp(self.t_embedding.weight))
        v_item = self.ii_gcn(self.image_original_adj, v_item_embeds)
        v_user = torch.sparse.mm(self.R, v_item)
        latent_v_item_embeds = torch.cat([v_user, v_item], dim=0)

        t_item = self.ii_gcn(self.text_original_adj, t_item_embeds)
        t_user = torch.sparse.mm(self.R, t_item)
        latent_t_item_embeds = torch.cat([t_user, t_item], dim=0) 
        return v_item, t_item, latent_v_item_embeds, latent_t_item_embeds
    

    def forward(self, co_adj, train=False):

        # Latent Preference Extraction
        latent_v_item, latent_t_item, latent_v_item_embeds, latent_t_item_embeds = self.latent_preference_extract()

        latent_v_ui_embeds = self.ui_gcn(co_adj, self.latent_v_user.weight, latent_v_item)  
        latent_t_ui_embeds = self.ui_gcn(co_adj, self.latent_t_user.weight, latent_t_item) 

        latent_embeds = (latent_v_ui_embeds + latent_t_ui_embeds) / 2

        item_embeds = self.raw_item_embedding.weight
        user_embeds = self.user_embedding.weight
        co_ui_embeds = self.ui_gcn(co_adj, user_embeds, item_embeds)   


        # Fine-Grained Preference Encode
        coarse_grained_embeds, fine_grained_image, fine_grained_text = self.fgpe.forward(latent_v_item_embeds, latent_t_item_embeds, co_ui_embeds)

        # Modality-Aware Decode
        visual_token = self.visual_token_embedding(self.visual_token_index)
        text_token = self.text_token_embedding(self.text_token_index )
        fine_grained_text, fine_grained_image = self.ma_decoder.modality_aware(fine_grained_text, text_token, fine_grained_image, visual_token)
        factors_enhancement_embeds = (fine_grained_image + fine_grained_text + coarse_grained_embeds) / 3


        all_embeds = co_ui_embeds + factors_enhancement_embeds

        if train:
            return all_embeds, (factors_enhancement_embeds, co_ui_embeds, latent_embeds), (latent_v_item_embeds, latent_t_item_embeds)

        return all_embeds
    

    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (self.sq_sum(users) + self.sq_sum(pos_items) + self.sq_sum(neg_items)) / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        reg_loss = self.reg_weight_1 * regularizer

        return mf_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        
        return torch.mean(cl_loss)

    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        embeds_1, embeds_2, embeds_3 = self.forward(self.co_ui_adj, train=True)
        u_embs, i_embs = torch.split(embeds_1, [self.n_users, self.n_items], dim=0)
        u_copy = u_embs[users]
        pos_i = i_embs[pos_items]
        neg_i = i_embs[neg_items]
        
        factors_enhancement_embeds, co_ui_embeds, latent_embeds = embeds_2

        # BPR loss
        bpr_loss, reg_loss_1 = self.bpr_loss(u_copy, pos_i, neg_i)


        # IA loss
        latent_v_item_embeds, latent_t_item_embeds = embeds_3
        ia_loss = self.ia_loss * self.intra_align(latent_v_item_embeds, latent_t_item_embeds)
        # EA loss
        enhancement_users, enhancement_items = torch.split(factors_enhancement_embeds, [self.n_users, self.n_items], dim=0)
        co_user, co_items = torch.split(co_ui_embeds, [self.n_users, self.n_items], dim=0)
        ea_loss = self.sc_loss * (self.InfoNCE(enhancement_users[users], co_user[users], self.sc_temp) + self.InfoNCE(enhancement_items[pos_items], co_items[pos_items], self.sc_temp))
        # SC loss
        sc_loss = ia_loss + ea_loss
        

        # MP loss
        latent_user, latent_items = torch.split(latent_embeds, [self.n_users, self.n_items], dim=0)
        c_loss = self.InfoNCE(latent_user[users], enhancement_users[users], self.mp_temp)
        n1_loss = self.perturbation_contrastive_loss(users, enhancement_users, self.mp_temp)
        n2_loss = self.perturbation_contrastive_loss(users, latent_user, self.mp_temp)
        mp_loss = self.mp_loss * (c_loss + n1_loss + n2_loss)
        
        reg_loss_2 = self.reg_weight_2 * self.sq_sum(latent_items[pos_items]) / self.batch_size
        reg_loss = reg_loss_1 + reg_loss_2
        
        return bpr_loss + sc_loss + mp_loss + reg_loss
    

    
    def perturbation_contrastive_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss
    
    def intra_align(self, embed1, embed2):
        emb1_var, emb1_mean = torch.var(embed1), torch.mean(embed1)
        emb2_var, emb2_mean = torch.var(embed2), torch.mean(embed2)
        
        ia_loss = (torch.abs(emb1_var - emb2_var) + torch.abs(emb1_mean - emb2_mean)).mean()
        
        return ia_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]

        all_embeds = self.forward(self.co_ui_adj)
        restore_user_e, restore_item_e = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    


class FGPEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.separate_coarse = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)

        self.v_aware = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        self.t_aware = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, latent_v_item_embeds, latent_t_item_embeds, co_ui_embeds):

        image_logits = self.separate_coarse(latent_v_item_embeds)
        text_logits = self.separate_coarse(latent_t_item_embeds)
        logits = torch.cat([image_logits, text_logits], dim=-1)
        weights = self.softmax(logits)
        image_weights, text_weights = torch.split(weights, 1, dim=-1)

        coarse_grained_embeds = ( image_weights * latent_v_item_embeds + text_weights * latent_t_item_embeds )

        fine_grained_image = torch.multiply(self.v_aware(co_ui_embeds), (latent_v_item_embeds - coarse_grained_embeds))
        fine_grained_text = torch.multiply(self.t_aware(co_ui_embeds), (latent_t_item_embeds - coarse_grained_embeds))

        return coarse_grained_embeds, fine_grained_image, fine_grained_text


class MA_Decoder(nn.Module):
    def __init__(self, input_dim_v=32, input_dim_t=4096, hidden_dim=64):
        super(MA_Decoder, self).__init__()
        self.token_proj_v = nn.Linear(input_dim_v, hidden_dim)  
        self.token_proj_t = nn.Linear(input_dim_t, hidden_dim)  

        self.ent_attn_v = nn.Linear(hidden_dim * 2, 1) 
        self.ent_attn_t = nn.Linear(hidden_dim * 2, 1) 

    def pad_or_truncate(self, token_embedding, target_shape):
        token_num, feat_dim = token_embedding.shape
        target_num = target_shape[0]

        if token_num < target_num:
            padding = torch.zeros((target_num - token_num, feat_dim), device=token_embedding.device, dtype=token_embedding.dtype)
            token_embedding = torch.cat([token_embedding, padding], dim=0)
        elif token_num > target_num:
            token_embedding = token_embedding[:target_num]
        return token_embedding

    def modality_aware(self, e_t, token_t, e_v, token_v):
        token_v_pooled = token_v.mean(dim=1)  
        token_t_pooled = token_t.mean(dim=1)  

        token_v_pooled = token_v_pooled.to(self.token_proj_v.weight.dtype)
        token_t_pooled = token_t_pooled.to(self.token_proj_t.weight.dtype)

        token_v_proj = self.token_proj_v(token_v_pooled)
        token_t_proj = self.token_proj_t(token_t_pooled)

        token_v_proj = self.pad_or_truncate(token_v_proj, e_v.detach().shape)  
        token_t_proj = self.pad_or_truncate(token_t_proj, e_t.detach().shape)  

        v = torch.cat((e_v, token_v_proj), dim=-1)  
        u_v = torch.tanh(v)

        t = torch.cat((e_t, token_t_proj), dim=-1)  
        u_t = torch.tanh(t)

        scores_v = self.ent_attn_v(u_v)  
        scores_t = self.ent_attn_t(u_t)  

        attention_weights_v = torch.nn.functional.normalize(torch.softmax(scores_v, dim=0))
        attention_weights_t = torch.nn.functional.normalize(torch.softmax(scores_t, dim=0)) 

        context_vectors_v = attention_weights_v * e_v  
        context_vectors_t = attention_weights_t * e_t  

        return context_vectors_t, context_vectors_v
