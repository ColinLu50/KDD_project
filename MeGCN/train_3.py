import os
import sys

import torch
import pickle

from MetaGCN_v3 import MetaGCN
from gcn_dataloader_v3 import GCNDataLoader
from data_generation_megcn_v3 import generate_one_hot
from evaluation_v3 import evaluation_

import os
import torch
import random
import pickle
from tqdm import tqdm



random.seed(0)

config = {
    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_actor': 8030,
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    # cuda setting
    'use_cuda': True,
    # model setting
    'inner': 5, # update time
    'lr': 5e-4,
    'local_lr': 5e-6,
    'batch_size': 32,
    'num_epoch': 100,
    # candidate selection
    # 'num_candidate': 20,
    'gcn_layer_number' : 5
}



def training(model_, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        model_.cuda()

    best_ever = -1

    training_set_size = len(total_dataset)
    model_.train()
    for epoch in range(num_epoch):
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)

        losses = []

        for i in range(num_batch):
            try:
                s_pair_batch = list(a[batch_size*i:batch_size*(i+1)])
                s_y_batch = list(b[batch_size*i:batch_size*(i+1)])
                q_pair_batch = list(c[batch_size*i:batch_size*(i+1)])
                q_y_batch = list(d[batch_size*i:batch_size*(i+1)])

            except IndexError:
                continue

            batch_data = (s_pair_batch, s_y_batch,
                          q_pair_batch, q_y_batch)

            batch_loss = model_.global_update(batch_data, config['inner'])
            losses.append(batch_loss)

        print(f'Epoch {epoch} Loss: {torch.stack(losses).mean(0)}')
        sys.stdout.flush()

        if model_save:
            cur_v = evaluation_(megcn, master_path, 'tmp', test_state="user_and_item_cold_state")
            if cur_v > best_ever:
                best_ever = cur_v
                # print('Save')
                print('Better value:', best_ever, 'Save!')
                torch.save(model_, model_filename)

if __name__ == "__main__":
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m_final"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        generate_one_hot(master_path) # preparing dataset. It needs about 22GB of your hard disk space.

    # training model.
    ml_dataset = GCNDataLoader(master_path)
    ml_dataset.getSparseGraph()

    megcn = MetaGCN(config, ml_dataset)
    model_filename = "{}/MetaGCN5-5_v5.pkl".format(master_path)

    print('============ Config ===============')
    for k in config:
        print(k, ':', config[k])


    # Load training dataset.
    training_set_size = ml_dataset.state_size['warm_state']

    support_pairs_list = []
    # support_features_list = []
    support_ys_list = []
    query_pairs_list = []
    # query_features_list = []
    query_ys_list = []

    for idx in range(training_set_size):
        support_pairs_list.append(pickle.load(open("{}/warm_state/supp_pairs_{}.pkl".format(master_path, idx), "rb")))
        # support_features_list.append(pickle.load(open("{}/warm_state/supp_f_{}.pkl".format(master_path, idx), "rb")))
        support_ys_list.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))

        query_pairs_list.append(pickle.load(open("{}/warm_state/query_pairs_{}.pkl".format(master_path, idx), "rb")))
        # query_features_list.append(pickle.load(open("{}/warm_state/query_f_{}.pkl".format(master_path, idx), "rb")))
        query_ys_list.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))


    total_dataset = list(
        zip(support_pairs_list, support_ys_list,
            query_pairs_list, query_ys_list)
    )
    del (support_pairs_list, support_ys_list,
            query_pairs_list, query_ys_list)

    training(megcn, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    # training(megcn, total_dataset, batch_size=config['batch_size'], num_epoch=2, model_save=True, model_filename=model_filename)

    evaluation_(megcn, master_path, 'megcn5-5_v5')
    # evaluation_(megcn, master_path, 'megcn_v3_infer')