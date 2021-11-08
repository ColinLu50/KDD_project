import os
import torch
import pickle

from MeLU_reptile import MeLU
from options import config
# from model_training import training
from data_generation import generate
from evaluation_melu import evaluation_
import random


def training_reptile(melu, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        melu.cuda()

    training_set_size = len(total_dataset)
    melu.train()
    for e_num in range(num_epoch):
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            melu.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'], 1 - e_num / num_epoch)

    if model_save:
        torch.save(melu, model_filename)

if __name__ == "__main__":
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about 22GB of your hard disk space.
        generate(master_path)

    # training model.
    melu = MeLU(config)
    model_filename = "{}/MeLU5_test_reptile.pkl".format(master_path)


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

    total_dataset = list(
        zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    training_reptile(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    # training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=1, model_save=True, model_filename=model_filename)
    evaluation_(melu, master_path, 'melu5_reptile_test1')
