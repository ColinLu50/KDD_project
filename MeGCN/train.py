import os
import torch
import pickle

from MetaGCN import MetaGCN
from gcn_dataloader import GCNDataLoader
from options import config
from tmp import training
from data_generation import generate
from evaluation import evaluation_


if __name__ == "__main__":
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m"
    # if not os.path.exists("{}/".format(master_path)):
    #     os.mkdir("{}/".format(master_path))
    #     # preparing dataset. It needs about 22GB of your hard disk space.
    #     generate(master_path)

    # training model.
    ml_dataset = GCNDataLoader(master_path)
    megcn = MetaGCN(config, ml_dataset)
    model_filename = "{}/test_MetaGCN.pkl".format(master_path)


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

    total_dataset = list(
        zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_uids_s, supp_iids_s, query_uids_s, query_iids_s))
    del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_uids_s, supp_iids_s, query_uids_s, query_iids_s)

    training(megcn, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    # training(megcn, total_dataset, batch_size=config['batch_size'], num_epoch=1, model_save=True, model_filename=model_filename)

    evaluation_(megcn, master_path, 'megcn_2_simple')