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
from options import config


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = rate_list.index(str(row['rate'])) # 6
    rate_feature = np.zeros((1, config['num_rate']))
    rate_feature[0, rate_idx] = 1

    genre_idx = np.zeros((1, 25))
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1

    director_idx = np.zeros((1, 2186))
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1

    actor_idx = np.zeros((1, 8030))
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    # return torch.cat((rate_feature, genre_idx, director_idx, actor_idx), 1)

    return np.concatenate([rate_feature, genre_idx, director_idx, actor_idx], axis=1)

def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    # gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    # age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    # occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    # zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()


    gender_idx = gender_list.index(str(row['gender']))
    gender_feature = np.zeros((1, config['num_gender']))
    gender_feature[0, gender_idx] = 1

    age_idx = age_list.index(str(row['age']))
    age_feature = np.zeros((1, config['num_age']))
    age_feature[0, age_idx] = 1

    occupation_idx = occupation_list.index(str(row['occupation_code']))
    occupation_feature = np.zeros((1, config['num_occupation']))
    occupation_feature[0, occupation_idx] = 1

    zip_idx = zipcode_list.index(str(row['zip'])[:5])
    zip_feature = np.zeros((1, config['num_zipcode']))
    zip_feature[0, zip_idx] = 1

    # return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)
    return np.concatenate((gender_feature, age_feature, occupation_feature, zip_feature), axis=1)

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def generate_one_hot(master_path):
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

    # hashmap for item information
    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    max_uid = -1
    max_mid = -1

    warm_uids = []
    warm_iids = []

    state_size = {}



    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())
        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)


            if u_id > max_uid:
                max_uid = u_id


            seen_item_num = len(dataset[str(u_id)])
            indices = list(range(seen_item_num))

            if seen_item_num < 13 or seen_item_num > 100:
                continue



            random.shuffle(indices)
            cur_item_list = np.array(dataset[str(u_id)])
            cur_y = np.array(dataset_y[str(u_id)])

            if state == 'warm_state':
                warm_uids.extend([u_id] * (seen_item_num - 10))
                warm_iids.extend(cur_item_list.astype(int)[indices[:-10]])

            support_pairs = []
            support_features = []
            for m_id in cur_item_list[indices[:-10]]:
                m_id = int(m_id)
                if m_id > max_mid:
                    max_mid = m_id
                tmp_f = (user_dict[u_id], movie_dict[m_id]) # combine embedding of movie info and user info
                # try:
                #     support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                # except:
                #     support_x_app = tmp_x_converted
                support_pairs.append(torch.tensor([[u_id, m_id]]))
                # support_features.append(tmp_f)

            support_pairs = torch.cat(support_pairs, dim=0)
            # support_features = torch.cat(support_features, dim=0)

            query_pairs = []
            query_features = []
            for m_id in cur_item_list[indices[-10:]]:
                m_id = int(m_id)
                if m_id > max_mid:
                    max_mid = m_id
                tmp_f = (user_dict[u_id], movie_dict[m_id]) # combine embedding of movie info and user info
                query_pairs.append(torch.tensor([[u_id, m_id]]))
                # query_features.append(tmp_f)
            query_pairs = torch.cat(query_pairs, dim=0)

            support_y_app = torch.FloatTensor(cur_y[indices[:-10]])
            query_y_app = torch.FloatTensor(cur_y[indices[-10:]])

            pickle.dump(support_pairs, open("{}/{}/supp_pairs_{}.pkl".format(master_path, state, idx), "wb"))
            # pickle.dump(support_features, open("{}/{}/supp_f_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_pairs, open("{}/{}/query_pairs_{}.pkl".format(master_path, state, idx), "wb"))
            # pickle.dump(query_features, open("{}/{}/query_f_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))

            # with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
            #     for m_id in tmp_x[indices[:-10]]:
            #         f.write("{}\t{}\n".format(u_id, m_id))
            # with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
            #     for m_id in tmp_x[indices[-10:]]:
            #         f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1

        state_size[state] = idx


    all_info = (max_uid, max_mid, warm_uids, warm_iids, state_size)
    pickle.dump(all_info, open("{}/all_info.pkl".format(master_path), "wb"))

    print('Generationg Done')




