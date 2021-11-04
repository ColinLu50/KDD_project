
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
from MetaGCN import MetaGCN
from options import config
from evaluation import evaluation_


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def generate(master_path):
    dataset_path = "movielens/ml-1m"
    rate_list = load_list("{}/m_rate.txt".format(dataset_path))
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    actor_list = load_list("{}/m_actor.txt".format(dataset_path))
    director_list = load_list("{}/m_director.txt".format(dataset_path))
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

    if not os.path.exists("{}/warm_state/".format(master_path)):
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    dataset = movielens_1m()

    movie_dict = {}
    for idx, row in dataset.item_data.iterrows():
        m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
        movie_dict[row['movie_id']] = m_info

    # hashmap for user profile
    user_dict = {}
    for idx, row in dataset.user_data.iterrows():
        u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
        user_dict[row['user_id']] = u_info
    pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))

    for state in states:
        idx = 0
        skip_num = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))

            if seen_movie_len < 13 or seen_movie_len > 100:
                skip_num += 1
                continue

            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])

            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1) # concatenate embedding of movie info and user info
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted

            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
            query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])



            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1

        print(f'{state} skip number {skip_num} / {len(dataset)}')
        # break

def training(model_, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        model_.cuda()

    training_set_size = len(total_dataset)
    model_.train()
    for _ in range(num_epoch):
        print('Epoch ', _)
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d,e,f,g,h = zip(*total_dataset)
        for i in tqdm(range(num_batch)):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])

                supp_uids = list(e[batch_size * i:batch_size * (i + 1)])
                supp_iids = list(f[batch_size * i:batch_size * (i + 1)])
                query_uids = list(g[batch_size * i:batch_size * (i + 1)])
                query_iids = list(h[batch_size * i:batch_size * (i + 1)])

            except IndexError:
                continue

            batch_data = (supp_xs, supp_ys, query_xs, query_ys,
                          supp_uids, supp_iids, query_uids, query_iids)

            model_.global_update_MAML(batch_data, config['inner'])

    if model_save:
        torch.save(model_.state_dict(), model_filename)

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






