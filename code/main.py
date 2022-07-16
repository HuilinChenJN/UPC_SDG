import world
import utils
from world import cprint
import torch
import numpy as np
import model
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = model.PureMF(world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
# 加载预训练模型
if world.LOAD:
    try:
        # pretrain_file_path = './emebdding/mf-gowalla-64.pth.tar'
        pretrain_file_path = './code/embedding/mf-Office-64.pth.tar'
        # pretrain_file_path = './emebdding/mf-Clothing-64.pth.tar'
        pretrain_dict = torch.load(pretrain_file_path, map_location=torch.device('cpu'))
        original_model_dict = Recmodel.state_dict()
        # load the embeedings with same scope
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in original_model_dict}
        original_model_dict.update(pretrained_dict)
        Recmodel.load_state_dict(original_model_dict)
        world.cprint(f"loaded User and Item embedding weights from {pretrain_file_path}")
    except FileNotFoundError:
        print(f"pretrain_file_path not exists, please check it")
        exit()
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-UPC-SDG"))
else:
    w = None
    world.cprint("not enable tensorflowboard")

best_recall = 0.
best_precision = 0.
best_ndcg = 0.
best_loss = 999.
count = 1
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            print(result)
        aver_loss, time_info, bpr_loss, similarity_loss, std_loss, aver_similarity = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss{aver_loss:.3f}-bpr{bpr_loss:.3f}"
              f"-similarity_loss:{similarity_loss:.3f}-std{std_loss:.3f}"
              f"-{time_info}-similarity{aver_similarity:.3f}")

        if best_loss > aver_loss:
            best_loss = aver_loss
            torch.save(Recmodel.state_dict(), weight_file)
            count = 1
            # print(best_loss)
        else:
            count += 1
        if count > 30:
            cprint("[Train END]")
            break

finally:
    if world.tensorboard:
        w.close()

Procedure.output_generative_data(dataset, Recmodel, weight_file)