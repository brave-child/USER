
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
from collections import Counter, defaultdict

class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(os.getcwd()+config['data_path'] + config['dataset'])
            # if file exist?
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'

        dataset_path = os.path.abspath(os.getcwd()+config['data_path'] + config['dataset'])

        visual_token_index, visual_key_mask = get_entity_visual_tokens(dataset_path)
        text_token_index, text_key_mask = get_entity_textual_tokens(dataset_path)

        self.visual_token_index = visual_token_index.cuda()
        self.text_token_index = text_token_index.cuda()
        self.visual_key_mask = visual_key_mask
        self.text_key_mask = text_key_mask

@torch.no_grad()
def load_ent_map(dataset_path):
    item_map = {} 

    csv_file = os.path.join(dataset_path, "i_id_mapping.csv")
    df = pd.read_csv(csv_file, delimiter="\t")  # 以 TAB 分隔符读取
    for _, row in df.iterrows():
        item_map[str(row["itemID"])] = int(row["itemID"])  # key=asin, value=itemID
    return item_map

@torch.no_grad()
def get_entity_visual_tokens(dataset_path, max_num=8, token_size=8192):
    tokenized_result = json.load(open(f"{dataset_path}/visual_tokens.json", "r"))
    entity_dict = load_ent_map(dataset_path)

    token_dict = defaultdict(list)
    for i in range(token_size + 1):
        token_dict[i] = []

    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)  # 取 8 个 token
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)

    entity_to_token = defaultdict(list)

    for token, entities in token_dict.items():
        for ent in entities:
            entity_to_token[entity_dict[ent]].append(token)

    entid_tokens = []
    ent_key_mask = []

    for i in range(len(entity_dict)):
        if i in entity_to_token:
            entid_tokens.append(entity_to_token[i][:max_num])  # 只取前 8 个 token
            ent_key_mask.append([False] * max_num)  # 全部有效
        else:
            entid_tokens.append([token_size - 1] * max_num)  # 全部填充 `token_size - 1`
            ent_key_mask.append([True] * max_num)  # 全部 mask

    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()

@torch.no_grad()
def get_entity_textual_tokens(dataset_path, max_num=8, token_size=4096):
    tokenized_result = json.load(open(f"{dataset_path}/text_tokens.json", "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset_path)

    for i in range(token_size + 1):
        token_dict[i] = []

    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)  # 取 8 个 token
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)

    entity_to_token = defaultdict(list)

    for token, entities in token_dict.items():
        for ent in entities:
            entity_to_token[entity_dict[ent]].append(token)

    entid_tokens = []
    ent_key_mask = []

    for i in range(len(entity_dict)):
        if i in entity_to_token:
            entid_tokens.append(entity_to_token[i][:max_num])  
            ent_key_mask.append([False] * max_num) 
        else:
            entid_tokens.append([token_size - 1] * max_num) 
            ent_key_mask.append([True] * max_num)  

    return torch.LongTensor(entid_tokens), torch.BoolTensor(ent_key_mask).cuda()