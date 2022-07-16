from torch import nn
import torch
import numpy as np
from SelfLoss import SimilarityMarginLoss
import torch.nn.functional as F
import world

class Similar(nn.Module):
    def __init__(self):
        super(Similar, self).__init__()

    # Calculate the similarity loss between Item and replaceable item to be replaced
    def calculate_similar_loss(self, replaceable_feature, original_feature, privacy_settings):
        raise NotImplementedError


class RegularSimilar(Similar):

    def __init__(self, latent_dim, userSimMax, userSimMin):
        super(RegularSimilar, self).__init__()
        self.latent_dim = latent_dim
        # cos
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # set the method of similarity loss
        self.similarity_loss = SimilarityMarginLoss()

        self.user_sim_max = userSimMax
        self.user_sim_min = userSimMin

        # torch.nn.L1Loss()
        # liner transformer
        self.user_item_feature = nn.Linear((2 * self.latent_dim) + 1, self.latent_dim)

    def generate_original_item_mask(self, replace_scores, item_ids):
        item_ids = torch.tensor(item_ids).cuda()
        # get the max length of proposal
        item_matrix = torch.arange(0, replace_scores.shape[1]).long().cuda()
        mask_expand = item_matrix.unsqueeze(0).expand(replace_scores.shape[0], replace_scores.shape[1])
        item_expand = item_ids.unsqueeze(1).expand_as(mask_expand)
        original_mask = (item_expand == mask_expand).int()
        cover_msk = (item_expand != mask_expand).int()
        return original_mask, cover_msk

    def get_replaceable_item_similarity(self, replaceable_items_feature, all_items, replace_probability):
        # Calculate the similarity of the selected item and the original item
        chunk_number = 20
        total_similarity_score = torch.Tensor([]).cuda()
        offset_number = all_items.shape[0] / chunk_number
        # print(all_items.shape)
        for i in range(chunk_number):
            start = int(np.floor(i * offset_number))
            end = int(np.floor((i+1) * offset_number))
            scores = torch.mm(replaceable_items_feature, all_items[start:end].T)
            total_similarity_score = torch.cat([total_similarity_score, scores], dim=-1)

        # correlation
        item_similarity = (total_similarity_score * replace_probability).sum(dim=-1)

        return total_similarity_score, item_similarity


    def choose_replaceable_item(self, need_replace, union_feature, all_items, privacy_settings):
        item_ids = need_replace[:, 1]
        # get the embeddings of orginal items
        items_emb = all_items[item_ids]
        # obtain a union feature with item, user and privacy setting
        union_feature = torch.cat([union_feature, privacy_settings.view(-1, 1)], dim=-1)
        user_item_feature = self.user_item_feature(union_feature)
        # get the score with all items
        replace_score = torch.mm(user_item_feature, all_items.T)
        # get a mask matrix to keep original index
        original_mask, cover_msk = self.generate_original_item_mask(replace_score, item_ids)
        # get valid  values
        replace_score = replace_score * cover_msk
        if world.is_train:
            # train step
            # get the one-hot index of replaced items.
            replace_probability = F.gumbel_softmax(replace_score, hard=True, dim=-1)
            item_sequence = torch.arange(0, all_items.shape[0]).view(1, -1).cuda()
            # get the index and embeedings of replaced item
            replaceable_items = (replace_probability * item_sequence).sum(dim=-1).long()
            replaceable_items_feature = torch.mm(replace_probability, all_items)

            total_similarity_score, item_similarity = \
                self.get_replaceable_item_similarity(replaceable_items_feature, all_items, original_mask)
            # normalization
            item_similarity = self.regularize_similarity(total_similarity_score, item_similarity)

            del total_similarity_score
            # Calculate the similarity loss of each vector with similarity threshold.
            similarity_loss = self.similarity_loss(item_similarity, privacy_settings)
            similarity = item_similarity.mean()
        else:
            # eval step
            replaceable_items = torch.argmax(replace_score, dim=1)
            replaceable_items_feature = all_items[replaceable_items]
            similarity_loss = 0.
            similarity = self.cos(replaceable_items_feature, items_emb).mean()

        return replaceable_items, replaceable_items_feature, similarity_loss, similarity

    def regularize_similarity(self, total_similarity_score, item_similarity):
        score_min = torch.min(total_similarity_score, dim=-1).values
        score_max = torch.max(total_similarity_score, dim=-1).values
        # normalization
        return (item_similarity - score_min) / (score_max - score_min)
