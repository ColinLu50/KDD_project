import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

# from embeddings import item, user


class GCN_Estimator(torch.nn.Module):
    def __init__(self, config, gcn_dataset):
        super(GCN_Estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']
        self.gcn_dataset = gcn_dataset



        # self.item_emb = ItemEmb(config, gcn_dataset.m_item)
        # self.user_emb = UserEmb(config, gcn_dataset.n_user)

        self.num_users = gcn_dataset.n_user
        self.num_items = gcn_dataset.m_item

        tmp_f = gcn_dataset.user_dict[gcn_dataset.warm_user_ids[0]]
        self.num_u_feature = len(tmp_f)
        self.user_feature_num_list = []
        for f_idx in range(self.num_u_feature):
            self.user_feature_num_list.append(tmp_f[f_idx].shape[1])

        tmp_f_2 = gcn_dataset.item_dict[gcn_dataset.warm_item_ids[0]]
        self.num_i_feature = len(tmp_f_2)
        self.item_feature_num_list = []
        for f_idx in range(self.num_i_feature):
            self.item_feature_num_list.append(tmp_f_2[f_idx].shape[1])



        self.user_embedding_dim = config['embedding_dim'] * self.num_u_feature
        self.item_embedding_dim = config['embedding_dim'] * self.num_i_feature

        # self.embedding_all_user = torch.nn.Embedding(
        #     num_embeddings=self.num_users, embedding_dim=self.user_embedding_dim).cuda()
        # self.embedding_all_item = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.item_embedding_dim).cuda()

        # torch.nn.init.normal_(self.embedding_user_relation.weight[gcn_dataset.n_user_train:, :], std=0.1)
        # torch.nn.init.normal_(self.embedding_item_relation.weight[:gcn_dataset.n_user_train, :], std=0.1)
        #
        # torch.nn.init.zeros_(self.embedding_user_relation.weight[:gcn_dataset.n_user_train, :])
        # torch.nn.init.zeros_(self.embedding_item_relation.weight[:gcn_dataset.n_user_train, :])

        # embeddings
        self.user_embeddings = []
        for f_num in self.user_feature_num_list:
            e_ = torch.nn.Linear(
                in_features=f_num,
                out_features=self.embedding_dim,
                bias=False
            )
            torch.nn.init.normal_(e_.weight, std=0.1)
            self.user_embeddings.append(e_)
        self.user_embeddings = torch.nn.ModuleList(self.user_embeddings)

        self.item_embeddings = []
        for f_num in self.item_feature_num_list:
            e_ = torch.nn.Linear(
                in_features=f_num,
                out_features=self.embedding_dim,
                bias=False
            )
            torch.nn.init.normal_(e_.weight, std=0.1)
            self.item_embeddings.append(e_)
        self.item_embeddings = torch.nn.ModuleList(self.item_embeddings)


        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

        self.gcn_layer_number = config['gcn_layer_number']

        self.build_features()

    def build_features(self):
        # warm_user_ids = set(self.gcn_dataset.warm_user_ids)
        # warm_item_ids = set(self.gcn_dataset.warm_item_ids)

        u_fs_list = [[] for _ in range(self.num_u_feature)]

        for u_id in range(self.gcn_dataset.n_user):
            if u_id in self.gcn_dataset.user_dict:
                cur_f = self.gcn_dataset.user_dict[u_id]
                for f_idx in range(self.num_u_feature):
                    u_fs_list[f_idx].append(cur_f[f_idx].float())
            else:
                for f_idx in range(self.num_u_feature):
                    u_fs_list[f_idx].append(torch.zeros((1, self.user_feature_num_list[f_idx]), dtype=torch.float))

        u_f_mask = []
        for f_idx in range(self.num_u_feature):
            cur_fs = torch.cat(u_fs_list[f_idx], dim=0)
            if self.use_cuda:
                cur_fs = cur_fs.cuda()
            u_f_mask.append(cur_fs)

        self.u_f_mask = u_f_mask

        # item

        i_fs_list = [[] for _ in range(self.num_i_feature)]

        for i_id in range(self.gcn_dataset.m_item):
            if i_id in self.gcn_dataset.item_dict:
                cur_f = self.gcn_dataset.item_dict[i_id]
                for f_idx in range(self.num_i_feature):
                    i_fs_list[f_idx].append(cur_f[f_idx].float())
            else:
                for f_idx in range(self.num_i_feature):
                    i_fs_list[f_idx].append(torch.zeros((1, self.item_feature_num_list[f_idx]), dtype=torch.float))

        i_f_mask = []
        for f_idx in range(self.num_i_feature):
            cur_fs = torch.cat(i_fs_list[f_idx], dim=0)
            if self.use_cuda:
                cur_fs = cur_fs.cuda()
            i_f_mask.append(cur_fs)

        self.i_f_mask = i_f_mask




    def computer_gcn(self, A_hat):
        # warm_user_ids = set(self.gcn_dataset.warm_user_ids)
        # warm_item_ids = set(self.gcn_dataset.warm_item_ids)
        #
        #
        # u_fs_list = [[] for _ in range(self.num_u_feature)]
        #
        # for u_id in warm_user_ids:
        #     cur_f = self.gcn_dataset.user_dict[u_id]
        #     for f_idx in range(self.num_u_feature):
        #         u_fs_list[f_idx].append(cur_f[f_idx])

        split_emb_all_users = []
        for f_idx in range(self.num_u_feature):
            cur_femb_all_users = self.user_embeddings[f_idx](self.u_f_mask[f_idx])
            split_emb_all_users.append(cur_femb_all_users)

        emb_all_users = torch.cat(split_emb_all_users, dim=1)

        # i_fs_list = [[] for _ in range(self.num_i_feature)]
        #
        # for i_id in warm_item_ids:
        #     cur_f = self.gcn_dataset.item_dict[i_id]
        #     for f_idx in range(self.num_i_feature):
        #         i_fs_list[f_idx].append(cur_f[f_idx])

        split_emb_all_items = []
        for f_idx in range(self.num_i_feature):
            # cur_fs = torch.cat(i_fs_list[f_idx], dim=0)
            cur_femb_all_items = self.item_embeddings[f_idx](self.i_f_mask[f_idx])
            split_emb_all_items.append(cur_femb_all_items)

        emb_all_items = torch.cat(split_emb_all_items, dim=1)

        all_emb = torch.cat([emb_all_users, emb_all_items])
        embs = [all_emb]

        g = A_hat

        for layer in range(self.gcn_layer_number):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users_gcn, items_gcn = torch.split(light_out, [self.num_users, self.num_items])
        return users_gcn, items_gcn




        # with torch.no_grad():
        #     for uid in warm_user_ids:
        #         self.embedding_all_user[uid] =
        #
        #     for iid in warm_item_ids:
        #         self.embedding_all_item[iid] = torch.nn.Parameter(torch.ones(self.embedding_dim).cuda())




    def forward(self, user_ids, item_ids, A_hat, training=True):
        # rate_idx = Variable(aux_info[:, 0], requires_grad=False)
        # genre_idx = Variable(aux_info[:, 1:26], requires_grad=False)
        # director_idx = Variable(aux_info[:, 26:2212], requires_grad=False)
        # actor_idx = Variable(aux_info[:, 2212:10242], requires_grad=False)
        # gender_idx = Variable(aux_info[:, 10242], requires_grad=False)
        # age_idx = Variable(aux_info[:, 10243], requires_grad=False)
        # occupation_idx = Variable(aux_info[:, 10244], requires_grad=False)
        # area_idx = Variable(aux_info[:, 10245], requires_grad=False)
        #
        # user_ids = Variable(user_ids, requires_grad=False)
        # item_ids = Variable(item_ids, requires_grad=False)
        #
        # item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx, item_ids)
        # user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx, user_ids)

        all_users_gcn, all_items_gcn = self.computer_gcn(A_hat)

        user_gcn_emb = all_users_gcn[user_ids]
        item_gcn_emb = all_items_gcn[item_ids]
        X = torch.cat((user_gcn_emb, item_gcn_emb), 1)


        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        return self.linear_out(X)




class MetaGCN(torch.nn.Module):
    def __init__(self, config, gcn_dataset):
        super(MetaGCN, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = GCN_Estimator(config, gcn_dataset)
        if self.use_cuda:
            self.model.cuda()


        self.local_lr = config['local_lr']
        self.meta_lr = config['lr']
        self.store_parameters()

        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']

        self.A_hat_train = gcn_dataset.getSparseGraph() # cache=False
        self.gcn_dataset = gcn_dataset


    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_y, support_pair_id, query_pair_id, num_local_update):
        support_uid, support_iid = support_pair_id[:, 0], support_pair_id[:, 1]
        query_uid, query_iid = query_pair_id[:, 0], query_pair_id[:, 1]

        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_uid, support_iid, self.A_hat_train)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.model(query_uid, query_iid, self.A_hat_train)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def inference(self, support_set_y, support_pair_id, query_pair_id, num_local_update):
        support_uid, support_iid = support_pair_id[:, 0], support_pair_id[:, 1]
        query_uid, query_iid = query_pair_id[:, 0], query_pair_id[:, 1]

        # build new hat
        new_A_hat = self.gcn_dataset.getNewSparseGraph(support_pair_id)


        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_uid, support_iid, new_A_hat)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.model(query_uid, query_iid, new_A_hat)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def local_update(self, support_set_y, support_pair_id, num_local_update):
        # SGD
        support_uid, support_iid = support_pair_id[:, 0], support_pair_id[:, 1]

        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_uid, support_iid, self.A_hat_train)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                # if self.weight_name[i] in self.local_update_target_weight_name:
                #     self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                # else:
                #     self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
                self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]



        return self.fast_weights

    def global_update(self, batch_data, num_local_update):
        # reptile

        (s_pair_batch, s_featur_batch, s_y_batch,
         q_pair_batch, q_featur_batch, q_y_batch) = batch_data

        batch_sz = len(s_pair_batch)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                s_pair_batch[i] = s_pair_batch[i].cuda()
                # s_featur_batch[i] = s_featur_batch[i].cuda()
                s_y_batch[i] = s_y_batch[i].cuda()

                q_pair_batch[i] = q_pair_batch[i].cuda()
                # q_featur_batch[i] = q_featur_batch[i].cuda()
                q_y_batch[i] = q_y_batch[i].cuda()

        new_weights = deepcopy(self.keep_weight)
        for i in range(batch_sz):
            local_weights = self.local_update(s_y_batch[i], s_pair_batch[i], num_local_update)
            for i in range(self.weight_len):
                weight_name_ = self.weight_name[i]
                new_weights[weight_name_] += (local_weights[weight_name_] - self.keep_weight[weight_name_]) * (
                            self.local_lr / batch_sz)

        self.model.load_state_dict(new_weights)
        self.store_parameters()
        return

    # def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
    #     tmp = 0.
    #     if self.cuda():
    #         support_set_x = support_set_x.cuda()
    #         support_set_y = support_set_y.cuda()
    #     for idx in range(num_local_update):
    #         if idx > 0:
    #             self.model.load_state_dict(self.fast_weights)
    #         weight_for_local_update = list(self.model.state_dict().values())
    #         support_set_y_pred = self.model(support_set_x)
    #         loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
    #         # unit loss
    #         loss /= torch.norm(loss).tolist()
    #         self.model.zero_grad()
    #         grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
    #         for i in range(self.weight_len):
    #             # For averaging Forbenius norm.
    #             tmp += torch.norm(grad[i])
    #             if self.weight_name[i] in self.local_update_target_weight_name:
    #                 self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
    #             else:
    #                 self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
    #     return tmp / num_local_update
