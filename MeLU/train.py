import os
import torch
import pickle

from MeLU import MeLU
from options import config
from model_training import training
from data_generation import generate
from evaluation import evaluation_


if __name__ == "__main__":
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about 22GB of your hard disk space.
        generate(master_path)

    # training model.
    melu = MeLU(config)
    model_filename = "{}/test.pkl".format(master_path)
    # Load training dataset.
    training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(training_set_size):
        supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
    training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)

    evaluation_(melu, master_path, 1)