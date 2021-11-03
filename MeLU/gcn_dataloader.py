import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import json
from tqdm import tqdm
import pickle

from options import states
from data_generation import load_list, item_converting, user_converting
from dataset import movielens_1m

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = 2048
config['latent_dim_rec'] = 64
config['lightGCN_n_layers']= 3
config['dropout'] = 0
config['keep_prob']  = 0.6
config['A_n_fold'] = 100
config['test_u_batch_size'] = 100
config['multicore'] = 0
config['lr'] = 0.001
config['decay'] = 1e-4
config['pretrain'] = 0
config['A_split'] = False
config['bigdata'] = False

#
# class BasicDataset(Dataset):
#     def __init__(self):
#         print("init dataset")
#
#     @property
#     def n_users(self):
#         raise NotImplementedError
#
#     @property
#     def m_items(self):
#         raise NotImplementedError
#
#     @property
#     def trainDataSize(self):
#         raise NotImplementedError
#
#     @property
#     def testDict(self):
#         raise NotImplementedError
#
#     @property
#     def allPos(self):
#         raise NotImplementedError
#
#     def getUserItemFeedback(self, users, items):
#         raise NotImplementedError
#
#     def getUserPosItems(self, users):
#         raise NotImplementedError
#
#     def getUserNegItems(self, users):
#         """
#         not necessary for large dataset
#         it's stupid to return all neg items in super large dataset
#         """
#         raise NotImplementedError
#
#     def getSparseGraph(self):
#         """
#         build a graph in torch.sparse.IntTensor.
#         Details in NGCF's matrix form
#         A =
#             |I,   R|
#             |R^T, I|
#         """
#         raise NotImplementedError


class Loader():
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, master_path):

        dataset_path = "movielens/ml-1m"
        # train or test
        print(f'loading from [{dataset_path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        # train_file = path + '/train.txt'
        # test_file = path + '/test.txt'
        # self.path = path
        # trainUniqueUsers, trainItem, trainUser = [], [], []
        # testUniqueUsers, testItem, testUser = [], [], []
        warm_uids = []
        warm_iids = []

        all_uids = []
        all_user_info_ids = []

        all_iids = []
        all_item_info_ids = []




        # self.state_idx2uid = {state: {} for state in states}
        dataset = movielens_1m()
        dataset_path = "movielens/ml-1m"
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        actor_list = load_list("{}/m_actor.txt".format(dataset_path))
        director_list = load_list("{}/m_director.txt".format(dataset_path))
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

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

        for state in states:
            idx = 0
            # if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            #     os.mkdir("{}/{}/{}".format(master_path, "log", state))
            with open("{}/{}.json".format(dataset_path, state), encoding="utf-8") as f:
                dataset = json.loads(f.read())
            with open("{}/{}_y.json".format(dataset_path, state), encoding="utf-8") as f:
                dataset_y = json.loads(f.read())

            for _, user_id in tqdm(enumerate(dataset.keys())):
                u_id = int(user_id)

                if u_id > max_uid:
                    max_uid = u_id

                seen_movie_len = len(dataset[str(u_id)])
                indices = list(range(seen_movie_len))

                if seen_movie_len < 13 or seen_movie_len > 100:
                    continue


                # trainUniqueUsers.append(u_id)
                # trainUser.extend([u_id] * len(items))
                # trainItem.extend(items)

                cur_m_ids = np.array(dataset[str(u_id)])
                tmp_y = np.array(dataset_y[str(u_id)])

                support_x_app = None
                for m_id in cur_m_ids:
                    m_id = int(m_id)
                    if m_id > max_mid:
                        max_mid = m_id

                if state == 'warm_state':
                    warm_uids.extend([u_id] * (len(cur_m_ids) - 10))
                    warm_iids.extend(cur_m_ids)


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

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    # trainUser[i] - trainItem[i] is relation pair
                    self.m_item = max(self.m_item, max(items)) # max item id for the partition
                    self.n_user = max(self.n_user, uid) # max user id for the partitioin
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        # user-item interaction matrix R (UserNumber x ItemNumber)
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1 # smooth?
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self ,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fol d *fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        # Coordinate matrix M,  M[row[i], col[i]] = M.data[i]
        # Covert to row, col, data and construct a new tensor by them
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # matrix A
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                # Matrix D^(-1/2)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {en d -s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        # posItems[uid] = [item id that connect to uid]
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems