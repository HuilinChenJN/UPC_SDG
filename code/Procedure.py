'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import collections
import numba as nb

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        # 采样副样本
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    total_similarity = 0.
    total_similarity_loss = 0.
    total_bpr = 0.
    total_std_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):

        if len(batch_users) < world.config['bpr_batch_size']:
            continue
        # 增加每个用户的正样本
        unique_user, pos_item_index, mask = load_users_pos_items(dataset, batch_users)
        # start_time = time()
        cri, bpr_loss, std_loss, similarity_loss, similarity = bpr.stageOne(batch_users, batch_pos, batch_neg, unique_user, pos_item_index, mask)
        # end_time = time()
        # print('计算时间', end_time - start_time)

        aver_loss += cri
        total_similarity += similarity
        total_similarity_loss += similarity_loss
        total_bpr += bpr_loss
        total_std_loss += std_loss
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    aver_similarity = total_similarity / total_batch
    aver_bpr = total_bpr / total_batch
    aver_std_loss= total_std_loss / total_batch
    aver_similarity_loss= total_similarity_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return aver_loss, time_info, aver_bpr, aver_similarity_loss, aver_std_loss, aver_similarity


def load_users_pos_items(dataset, batch_users):
    batch_users = batch_users.detach().cpu().numpy()
    unique_user = list(set(batch_users))
    all_pos_list = np.array(dataset.allPos, dtype=object)
    users_all_pos_items = all_pos_list[unique_user]
    lens = [len(item) for item in users_all_pos_items]
    max_pos_len = max(lens)
    pos_item_index, mask = normal_users_pos_list(users_all_pos_items, max_pos_len)
    return unique_user, pos_item_index, mask


@nb.jit(forceobj=True)
def normal_users_pos_list(users_all_pos_items, max_len):
    mask = np.zeros((len(users_all_pos_items), max_len))
    pos_item_index = np.zeros((len(users_all_pos_items), max_len))
    for index, item in enumerate(users_all_pos_items):
        user_pos_num = len(item)
        pos_item_index[index] = np.concatenate((item, np.zeros(max_len-user_pos_num)))
        mask[index] = np.concatenate((np.ones(user_pos_num), np.zeros(max_len-user_pos_num)))
    return pos_item_index, mask


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        # print(results)
        return results


def output_generative_data(dataset, recommend_model, weight_file):
    # 加载最优的模型数据
    recommend_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    recommend_model.eval()
    print(f"loaded model best weights from {weight_file}")
    # 数据集中所有用户的item列表
    all_pos_list = np.array(dataset.allPos, dtype=object)
    # 生成数据集
    users = np.arange(0, dataset.n_users)
    world.is_train = False
    # 循环获取每个用户要替换的数据和被替换的数据信息
    output_file_name = './output/{}-replace{}-privacy{}.txt'\
        .format(world.dataset, world.config['replace_ratio'], world.config['privacy_ratio'])
    total_similarity = 0.
    with open(output_file_name, 'w+') as f:
        for user_id in users:
            users_all_pos_items = all_pos_list[user_id]
            pos_item_index = np.array(users_all_pos_items.tolist())
            user_pos_items = np.array([pos_item_index])
            train_pos = torch.tensor([pos_item_index])
            mask = np.array([np.ones(len(pos_item_index))])
            unique_user = [user_id]
            need_replace, replaceable_items, replaceable_items_feature, feature_loss = \
                recommend_model.computer_pos_score(unique_user, user_pos_items, mask, train_pos)
            if need_replace.shape[0] == 0:
                pos_item_index = pos_item_index.astype(np.str)
            else:
                original_items = need_replace[:, 1]
                for iter_id, original_item in enumerate(original_items):
                    item_index = np.argwhere(pos_item_index == original_item)[0][0]
                    pos_item_index[item_index] = replaceable_items[iter_id]
                pos_item_index = pos_item_index.astype(np.str)
            out_str = str(user_id) + ' ' + ' '.join(pos_item_index.tolist())+'\n'
            f.write(out_str)
    world.cprint(f"output new train is successful, save path is {output_file_name}")
    print(f"Average similarity is {total_similarity/len(users)}")

