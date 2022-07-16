'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import json
import numba as nb
import os

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.bpr_loss_d = config['bpr_loss_d']
        self.similarity_loss_d = config['similarity_loss_d']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, unique_user, pos_item_index, pos_item_mask):
        # start_time = time()
        CF1_loss, CF1_reg_loss, CF2_loss, similarity_loss, similarity, std_loss = self.model.bpr_loss(
            users, pos, neg, unique_user, pos_item_index, pos_item_mask)
        reg_loss = CF1_reg_loss*self.weight_decay
        # print(loss, reg_loss, similarity_loss)
        loss = self.bpr_loss_d * CF2_loss + self.similarity_loss_d * similarity_loss + std_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item(), CF2_loss.cpu().item(), std_loss.cpu().item(), similarity_loss.cpu().item(), similarity


@nb.jit(nopython=True)
def replace_original_to_replaceable(users, pos_items, need_replace):
    pos_mask = np.zeros(len(pos_items))
    replaceable_mask = []
    for array_index, user_item in enumerate(need_replace):
        user_index = user_item[0]
        item_index = user_item[1]
        user_index_array = np.nonzero(np.asfarray(users == user_index))[0]
        item_index_array = np.nonzero(np.asfarray(pos_items == item_index))[0]
        if len(item_index_array) > 0:
            intersect = np.intersect1d(user_index_array, item_index_array)
            if len(intersect) > 0:
                for index in intersect:
                    # 哪些需要被替换
                    pos_mask[index] = 1
                    # 哪些是可以被用于替换
                    replaceable_mask.append(array_index)
    replaceable_mask = np.array(replaceable_mask)
    return pos_mask, replaceable_mask


@nb.jit(nopython=True)
def construct_need_replace_user_item(users, sorted_pos_score, sorted_pos_index,
                                     pos_item_index, replace_ratio, train_pos):
    need_replace = []
    for user_id, item_score in enumerate(sorted_pos_score):
        user_index = users[user_id]
        # 获取当前用户的所有选取概率大于0的元素
        user_item_sorted_index = sorted_pos_index[user_id][item_score > 0]
        # 根据索引取出所有有效的item的得分排名
        valid_pos_item_list = pos_item_index[user_id][user_item_sorted_index]
        # 根据阈值计算 要替换的item的索引位置
        # attention 按照倒序排列选取尾端的数据
        need_replace_item_start = len(valid_pos_item_list) - round(len(valid_pos_item_list) * replace_ratio)
        need_replace_items = valid_pos_item_list[need_replace_item_start:]
        for item_id in need_replace_items:
            if item_id in train_pos:
                need_replace.append([user_index, item_id])
    return need_replace


def generate_user_privacy_settings(dataset_name, privacy_ration, replace_ration=0, replace_value=0):
    # 读取训练数据
    user_idxes = []
    training_data_path = os.path.join('./data/', dataset_name)
    with open(os.path.join(training_data_path, 'train.txt')) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                user_idxes.append(uid)

    max_user_id = max(user_idxes)
    user_ids = np.array([i for i in range(0, max_user_id + 1)], dtype=np.int64)

    user_idxes = torch.tensor(user_idxes, dtype=torch.int64)
    user_id_sequence = torch.from_numpy(user_ids)

    index_mask = torch.zeros_like(user_id_sequence).bool()
    need_different_index = torch.zeros_like(user_id_sequence)
    user_privacy_rationes = torch.zeros_like(user_id_sequence, dtype=torch.float64)
    # 赋值
    index_mask[user_idxes] = 1

    np.random.shuffle(user_ids)

    end_index = int(np.round(user_ids.shape[0] * replace_ration))
    need_different_index[user_ids[:end_index]] = 1
    need_different_index = need_different_index.bool()
    user_privacy_rationes = torch.masked_fill(user_privacy_rationes, index_mask, privacy_ration)
    user_privacy_rationes = torch.masked_fill(user_privacy_rationes, need_different_index, replace_value)

    user_privacy_rationes = user_privacy_rationes.numpy().tolist()

    # print(user_privacy_rationes)

    with open(os.path.join(training_data_path, 'user_privacy.json'), "w+") as out:
        json.dump(user_privacy_rationes, out)
    print('user_privacy dict output is successful, similarity:{0}, random replace:{1}, replace:{2}'.format(
        privacy_ration, replace_ration, replace_value
    ))
    return os.path.join(training_data_path, 'user_privacy.json')


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    # 根据用户 生成一个随机数
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    file = f"{world.dataset}-mf-{world.output_prefix}-" \
           f"{world.config['replace_ratio']}-{world.config['privacy_ratio']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
