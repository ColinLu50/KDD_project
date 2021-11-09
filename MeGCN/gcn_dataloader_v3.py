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

config['user_feature_num'] = 2 + 7 + 21 + 3402
config['item_feature_num'] = 6 + 25 + 2186 + 8030


class GCNDataLoader():
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, master_path):

        self.path = master_path

        dataset_path = "movielens/ml-1m"
        # train or test
        print(f'loading from [{dataset_path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']


        all_info = pickle.load(open("{}/all_info.pkl".format(master_path), "rb"))
        max_uid, max_mid, warm_uids, warm_iids, state_size = all_info
        self.n_user = max_uid + 1
        self.m_item = max_mid + 1
        self.warm_user_ids = warm_uids
        self.warm_item_ids = warm_iids
        self.state_size = state_size

        self.num_user_feature = config['user_feature_num']
        self.num_item_feature = config['item_feature_num']

        self.item_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

        self.Graph = None
        print(f"{len(warm_iids)} interactions for training")

        # user - feature of user
        u_uf_net = np.zeros((self.n_user, config['user_feature_num']))
        for u_id in self.user_dict:
            u_fearue = self.user_dict[u_id]
            u_uf_net[u_id, :] = u_fearue.reshape(-1)
        self.u_uf_net = csr_matrix(u_uf_net)

        # item - feature of items
        i_if_net = np.zeros((self.m_item, config['item_feature_num']))
        for i_id in self.item_dict:
            i_fearue = self.item_dict[i_id]
            i_if_net[i_id, :] = i_fearue.reshape(-1)

        self.i_fi_net = csr_matrix(i_if_net)

        # (users,items), bipartite graph
        # user-item interaction matrix R (UserNumber x ItemNumber)
        self.u_i_net = csr_matrix((np.ones(len(warm_iids)), (warm_uids, warm_iids)),
                                      shape=(self.n_user, self.m_item))


        users_items_D = np.array(self.u_i_net.sum(axis=1)).squeeze()
        users_ufs_D = np.array(self.u_uf_net.sum(axis=1)).squeeze()
        self.users_D = (users_items_D + users_ufs_D).astype(np.float32)
        # self.users_D[self.users_D == 0.] = 1

        items_users_D = np.array(self.u_i_net.sum(axis=0)).squeeze()
        items_ifs_D = np.array(self.i_fi_net.sum(axis=1)).squeeze()
        self.items_D = (items_users_D + items_ifs_D).astype(np.float32)
        # self.items_D[self.items_D == .0] = 1.

        self.ufs_D = np.array(self.u_uf_net.sum(axis=0)).squeeze().astype(np.float32)
        # self.ufs_D[self.ufs_D == 0.] = 1.

        self.ifs_D = np.array(self.i_fi_net.sum(axis=0)).squeeze().astype(np.float32)
        # self.ifs_D[self.ifs_D == 0.] = 1.


        # self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        # self.users_D[self.users_D == 0.] = 1
        # self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        # self.items_D[self.items_D == 0.] = 1.

        print(f"dataset is ready to go")


    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    # @property
    # def trainDataSize(self):
    #     return self.traindataSize
    #
    # @property
    # def testDict(self):
    #     return self.__testDict
    #
    # @property
    # def allPos(self):
    #     return self._allPos

    # def _split_A_hat(self ,A):
    #     A_fold = []
    #     fold_len = (self.n_users + self.m_items) // self.folds
    #     for i_fold in range(self.folds):
    #         start = i_fol d *fold_len
    #         if i_fold == self.folds - 1:
    #             end = self.n_users + self.m_items
    #         else:
    #             end = (i_fold + 1) * fold_len
    #         A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
    #     return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        # Coordinate matrix M,  M[row[i], col[i]] = M.data[i]
        # Covert to row, col, data and construct a new tensor by them
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getNewSparseGraph(self, new_u_i_pairs):
        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        update_new_uids = []
        update_new_iids = []
        for u_i_pair in new_u_i_pairs:
            u_id, i_id = u_i_pair
            if self.UserItemNet[u_id.cpu().numpy(), i_id.cpu().numpy()] == 0:
                update_new_uids.append(u_id)
                update_new_iids.append(i_id)

        AddUserItemNet = csr_matrix((np.ones(len(update_new_iids)), (update_new_uids, update_new_iids)),
                                 shape=(self.n_user, self.m_item))

        R = self.UserItemNet + AddUserItemNet
        R = R.tolil()
        # csr_matrix((np.ones(len(warm_iids)), (warm_uids, warm_iids)),
        #            shape=(self.n_user, self.m_item))
        # new_norm_


        # norm_adj = self.norm_adj

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        # matrix A

        rowsum = np.array(adj_mat.sum(axis=1))
        with np.errstate(divide='ignore'):
            d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        # Matrix D^(-1/2)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        new_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        return new_graph.coalesce().cuda()



    def getSparseGraph_old(self, cache=True):
        print("loading adjacency matrix")
        graph_save_path = os.path.join(self.path, 's_pre_adj_mat.npz')
        if self.Graph is None:
            try:
                if not cache:
                    raise Exception()
                pre_adj_mat = sp.load_npz(graph_save_path)
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

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                # Matrix D^(-1/2)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end -s}s, saved norm_mat...")
                sp.save_npz(graph_save_path, norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().cuda()
                print("don't split the matrix")
        return self.Graph



    def getSparseGraph(self, cache=True):
        print("loading adjacency matrix")
        graph_save_path = os.path.join(self.path, 's_pre_adj_mat.npz')
        if self.Graph is None:
            try:
                if not cache:
                    raise Exception()
                pre_adj_mat = sp.load_npz(graph_save_path)
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                L = self.n_users + self.m_items + self.num_user_feature + self.num_item_feature
                adj_mat = sp.dok_matrix((L, L), dtype=np.float32)
                adj_mat = adj_mat.tolil()

                # User - Item Graph
                A_u_i = self.u_i_net.tolil()
                adj_mat[:self.n_users, self.n_users: self.n_users + self.m_items] = A_u_i
                adj_mat[self.n_users: self.n_users + self.m_items, :self.n_users] = A_u_i.T

                # User - Feature of User Graph
                A_u_fu = self.u_uf_net.tolil()
                adj_mat[:self.n_users,
                (self.n_users + self.m_items):
                (self.n_users + self.m_items + self.num_user_feature)] = A_u_fu
                adj_mat[(self.n_users + self.m_items): (self.n_users + self.m_items + self.num_user_feature),
                :self.n_users] = A_u_fu.T

                # Item - Feature of Item  Graph
                A_i_fi = self.i_fi_net.tolil()
                idx_1 = self.n_users
                idx_2 = self.n_users + self.m_items
                idx_3 = self.n_users + self.m_items + self.num_user_feature
                adj_mat[idx_1: idx_2, idx_3:] = A_i_fi
                adj_mat[idx_3:, idx_1: idx_2] = A_i_fi.T

                adj_mat = adj_mat.todok()
                # matrix A

                rowsum = np.concatenate([self.users_D, self.items_D, self.ufs_D, self.ifs_D], axis=0)
                # rowsum = np.array(adj_mat.sum(axis=1))

                # diff = rowsum_test - rowsum.flatten()
                # print(np.max(diff))

                d_inv = np.power(rowsum, -0.5)
                # assert d_inv == np.power(rowsum_test, -0.5)
                d_inv[np.isinf(d_inv)] = 0.

                d_mat = sp.diags(d_inv)
                # Matrix D^(-1/2)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end -s}s, saved norm_mat...")
                sp.save_npz(graph_save_path, norm_adj)

            # if self.split == True:
            #     self.Graph = self._split_A_hat(norm_adj)
            #     print("done split matrix")
            # else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().cuda()
            print("don't split the matrix")
        return self.Graph