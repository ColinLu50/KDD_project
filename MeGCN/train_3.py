import os
import torch
import pickle

from MetaGCN_v3 import MetaGCN
from gcn_dataloader_v3 import GCNDataLoader
from data_generation_megcn_v3 import generate_one_hot
from evaluation_v2 import evaluation_

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
    'inner': 10, # update time
    'lr': 5e-5,
    'local_lr': 1e-6,
    'batch_size': 32,
    'num_epoch': 20,
    # candidate selection
    'num_candidate': 20,
    'gcn_layer_number' : 3
}



def training(model_, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        model_.cuda()

    training_set_size = len(total_dataset)
    model_.train()
    for _ in range(num_epoch):
        print('Epoch ', _)
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d,e,f = zip(*total_dataset)
        for i in tqdm(range(num_batch)):
            try:
                s_pair_batch = list(a[batch_size*i:batch_size*(i+1)])
                s_featur_batch = list(b[batch_size*i:batch_size*(i+1)])
                s_y_batch = list(c[batch_size*i:batch_size*(i+1)])


                q_pair_batch = list(d[batch_size*i:batch_size*(i+1)])
                q_featur_batch = list(e[batch_size * i:batch_size * (i + 1)])
                q_y_batch = list(f[batch_size * i:batch_size * (i + 1)])

            except IndexError:
                continue

            batch_data = (s_pair_batch, s_featur_batch, s_y_batch,
                          q_pair_batch, q_featur_batch, q_y_batch)

            model_.global_update(batch_data, config['inner'])

    if model_save:
        torch.save(model_, model_filename)

if __name__ == "__main__":
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m_final"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        generate_one_hot(master_path) # preparing dataset. It needs about 22GB of your hard disk space.

    # training model.
    ml_dataset = GCNDataLoader(master_path)
    # ml_dataset.getNewSparseGraph(torch.Tensor([[0, 0], [4, 21]]))
    exit(0)

    megcn = MetaGCN(config, ml_dataset)
    model_filename = "{}/MetaGCN_v3_seperate.pkl".format(master_path)

    # Load training dataset.
    training_set_size = ml_dataset.state_size['warm_state']

    support_pairs_list = []
    support_features_list = []
    support_ys_list = []
    query_pairs_list = []
    query_features_list = []
    query_ys_list = []

    for idx in range(training_set_size):
        support_pairs_list.append(pickle.load(open("{}/warm_state/supp_pairs_{}.pkl".format(master_path, idx), "rb")))
        support_features_list.append(pickle.load(open("{}/warm_state/supp_f_{}.pkl".format(master_path, idx), "rb")))
        support_ys_list.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))

        query_pairs_list.append(pickle.load(open("{}/warm_state/query_pairs_{}.pkl".format(master_path, idx), "rb")))
        query_features_list.append(pickle.load(open("{}/warm_state/query_f_{}.pkl".format(master_path, idx), "rb")))
        query_ys_list.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))


    total_dataset = list(
        zip(support_pairs_list, support_features_list, support_ys_list,
            query_pairs_list, query_features_list,query_ys_list)
    )
    del (support_pairs_list, support_features_list, support_ys_list,
            query_pairs_list, query_features_list,query_ys_list)

    training(megcn, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    # training(megcn, total_dataset, batch_size=config['batch_size'], num_epoch=1, model_save=True, model_filename=model_filename)

    evaluation_(megcn, master_path, 'megcn_v3')
    evaluation_(megcn, master_path, 'megcn_v3_infer')