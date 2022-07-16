from torch import nn
import torch
import numpy as np
from SelfLoss import SimilarityMarginLoss
import torch.nn.functional as F
import world

class Similar(nn.Module):
    def __init__(self):
        super(Similar, self).__init__()

    # 计算要替换的item和可替换item之间的相似度和loss
    def calculate_similar_loss(self, replaceable_feature, original_feature, privacy_settings):
        raise NotImplementedError

    # # 根据用户和item对形成新的特征X 并使用这个特征 给每个要替换的item 选择可以替换的item
    # def choose_replaceable_item(self, user_item_id, item_feature, all_items, privacy_settings):
    #     raise NotImplementedError


# 针对计算结果进行正规划
class RegularSimilar(Similar):

    def __init__(self, latent_dim, userSimMax, userSimMin):
        super(RegularSimilar, self).__init__()
        self.latent_dim = latent_dim
        # 计算相相似度
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # 设置损失计算
        self.similarity_loss = SimilarityMarginLoss()

        self.user_sim_max = userSimMax
        self.user_sim_min = userSimMin

        # torch.nn.L1Loss()
        # 设置线性变换
        self.user_item_feature = nn.Linear((2 * self.latent_dim) + 1, self.latent_dim)

    # 用来生成原始item的遮挡mask
    def generate_original_item_mask(self, replace_scores, item_ids):
        item_ids = torch.tensor(item_ids).cuda()
        # 获取最长的一个proposal的长度
        item_matrix = torch.arange(0, replace_scores.shape[1]).long().cuda()
        mask_expand = item_matrix.unsqueeze(0).expand(replace_scores.shape[0], replace_scores.shape[1])
        item_expand = item_ids.unsqueeze(1).expand_as(mask_expand)
        original_mask = (item_expand == mask_expand).int()
        cover_msk = (item_expand != mask_expand).int()
        return original_mask, cover_msk

    def get_replaceable_item_similarity(self, replaceable_items_feature, all_items, replace_probability):
        # 下面计算新选择的item和原始item的相似度
        # 计算选择出来的替换item和所有item的相似度
        chunk_number = 20
        total_similarity_score = torch.Tensor([]).cuda()
        offset_number = all_items.shape[0] / chunk_number
        # print(all_items.shape)
        for i in range(chunk_number):
            start = int(np.floor(i * offset_number))
            end = int(np.floor((i+1) * offset_number))
            scores = torch.mm(replaceable_items_feature, all_items[start:end].T)
            total_similarity_score = torch.cat([total_similarity_score, scores], dim=-1)
        # replaceable_items_norm = torch.sqrt(
        #     torch.sum(replaceable_items_feature * replaceable_items_feature, dim=-1)).view(-1, 1)
        #
        # all_items_norm = torch.sqrt(torch.sum(all_items * all_items, dim=-1)).view(1, -1)
        #
        # sim_norm = torch.mm(replaceable_items_norm, all_items_norm)

        # total_similarity_score = total_similarity_score / sim_norm

        # 把item本身去掉
        # total_similarity_score = total_similarity_score * mask
        item_similarity = (total_similarity_score * replace_probability).sum(dim=-1)

        return total_similarity_score, item_similarity


    def choose_replaceable_item(self, need_replace, union_feature, all_items, privacy_settings):
        item_ids = need_replace[:, 1]
        # 原始的item特征
        items_emb = all_items[item_ids]
        # 基于用户和item的联合特征 生成一个新的特征Z
        union_feature = torch.cat([union_feature, privacy_settings.view(-1, 1)], dim=-1)
        user_item_feature = self.user_item_feature(union_feature)
        # 计算新特征和所有采样item的得分
        replace_score = torch.mm(user_item_feature, all_items.T)
        # 获得原始item的位置mask
        original_mask, cover_msk = self.generate_original_item_mask(replace_score, item_ids)
        # 把原始的item得分进行遮挡
        # # 遮挡住原先的item得分
        replace_score = replace_score * cover_msk
        # 采用得分最高的那个元素用于替换
        if world.is_train:
            # 从联合feature和原始item之间选择一个得分最高的item出来
            # 并且这个item的协同过滤结果不会变差
            replace_probability = F.gumbel_softmax(replace_score, hard=True, dim=-1)
            # replace_probability = F.softmax(replace_score, dim=-1)

            item_sequence = torch.arange(0, all_items.shape[0]).view(1, -1).cuda()
            # 获得新item的编号
            replaceable_items = (replace_probability * item_sequence).sum(dim=-1).long()
            # 获得新的item的特征信息
            replaceable_items_feature = torch.mm(replace_probability, all_items)

            total_similarity_score, item_similarity = \
                self.get_replaceable_item_similarity(replaceable_items_feature, all_items, original_mask)
            # 进行一个归一化
            item_similarity = self.regularize_similarity(total_similarity_score, item_similarity)

            del total_similarity_score
            # 计算每个向量的相似度和相似度阈值的loss
            similarity_loss = self.similarity_loss(item_similarity, privacy_settings)
            # 平均相似度值
            similarity = item_similarity.mean()
            # 计算一下绝对相似度多少
            # replaceable_similarity = self.cos(replaceable_items_feature, items_emb)

            # print(similarity, replaceable_similarity.mean())

        else:
            replaceable_items = torch.argmax(replace_score, dim=1)
            replaceable_items_feature = all_items[replaceable_items]
            similarity_loss = 0.
            similarity = self.cos(replaceable_items_feature, items_emb).mean()

        return replaceable_items, replaceable_items_feature, similarity_loss, similarity

    def regularize_similarity(self, total_similarity_score, item_similarity):
        score_min = torch.min(total_similarity_score, dim=-1).values
        score_max = torch.max(total_similarity_score, dim=-1).values
        # 记录最大值和最小值
        # self.user_sim_max.append(score_max.mean())
        # self.user_sim_min.append(score_min.mean())

        # print(score_max.mean())

        return (item_similarity - score_min) / (score_max - score_min)
