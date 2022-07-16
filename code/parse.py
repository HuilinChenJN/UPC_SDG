'''
Created on Mar 1, 2020
Pytorch Implementation of UPC-SDG in

@author: Huilin Chen (clownclumsy@outlook.com)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go UPC-SDG")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [Office, Clothing, Gowalla]")
    parser.add_argument('--path', type=str,default="./save",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--replace_ratio', type=float, default=0.2,
                        help='set the ratio of needing to replace private item')
    parser.add_argument('--privacy_ratio', type=float,
                        help='set the ratio of user\'s privacy sensitivity')
    parser.add_argument('--privacy_settings_json', type=str,
                        help="the file name of user's privacy settings")
    # parser.add_argument('--coefficient', nargs='?', default="[1, 3]",
    #                     help='the number of sample from unvisited items')
    parser.add_argument('--bpr_loss_d', type=int, default=1,
                        help='the weight decay for bpr_loss')
    parser.add_argument('--similarity_loss_d', type=float, default=3,
                        help='the weight decay for similarity_loss')
    return parser.parse_args()
