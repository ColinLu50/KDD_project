
import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm

from options import states
from dataset import movielens_1m
from gcn_dataloader import GCNDataLoader
from MetaGCN_v2 import MetaGCN
from options import config
from evaluation import evaluation_





if __name__ == "__main__":
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m"
    # if not os.path.exists("{}/".format(master_path)):
    #     os.mkdir("{}/".format(master_path))
    #     # preparing dataset. It needs about 22GB of your hard disk space.

    # d1 = json.load(open('./movielens/ml-1m/user_cold_state.json', 'r'))
    # d2 = json.load(open('./movielens/ml-1m/warm_state.json', 'r'))
    #
    # items1 = set()
    # for k1 in d1:
    #     items1 = items1.union(set(d1[k1]))
    #
    # items2 = set()
    # for k2 in d2:
    #     items2 = items2.union(set(d2[k2]))
    #
    # print(len(items1))
    # print(len(items2))
    #
    # print(len(items2 & items1))
    ml_dataset = GCNDataLoader(master_path)
    # g = d.getSparseGraph()
    # print(g)

    master_path = "/home/workspace/big_data/KDD_projects_data/ml1m"
    # if not os.path.exists("{}/".format(master_path)):
    #     os.mkdir("{}/".format(master_path))
    #     # preparing dataset. It needs about 22GB of your hard disk space.
    #     generate(master_path)

    # training model.
    m = MetaGCN(config, ml_dataset)

    A = m.update_A_hat()

    exit(0)

    model_filename = "{}/test_1.pkl".format(master_path)
    # Load training dataset.
    training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    supp_uids_s = []
    supp_iids_s = []
    query_uids_s = []
    query_iids_s = []

    for idx in range(training_set_size):

        supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))

        _all_ids = pickle.load(open("{}/gcn_warm_state/ids_{}.pkl".format(master_path, idx), "rb"))
        supp_uids_s.append(_all_ids[0])
        supp_iids_s.append(_all_ids[1])
        query_uids_s.append(_all_ids[2])
        query_iids_s.append(_all_ids[3])


    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_uids_s, supp_iids_s, query_uids_s, query_iids_s))
    del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_uids_s, supp_iids_s, query_uids_s, query_iids_s)


    training(m, total_dataset, batch_size=config['batch_size'], num_epoch=2, model_save=True,
             model_filename=model_filename)

    # evaluation_(m, master_path, 1)

    # for i in range(len(supp_ys_s)):

    # a = m.model.computer_gcn()






