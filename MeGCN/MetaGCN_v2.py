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
        # GCN feature embedding
        self.num_users = gcn_dataset.n_user
        self.num_items = gcn_dataset.m_item


        # self.user_gcn_emb = torch.nn.Linear(
        #     in_features=self.num_users,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )

        # self.item_gcn_emb = torch.nn.Linear(
        #     in_features=self.num_items,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )
        self.user_gcn_emb = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim
        )
        self.item_gcn_emb = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim
        )




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

        # content-base feature embeddings
        self.user_feature_embeddings = []
        for f_num in self.user_feature_num_list:
            e_ = torch.nn.Linear(
                in_features=f_num,
                out_features=self.embedding_dim,
                bias=False
            )
            # torch.nn.init.normal_(e_.weight, std=0.1)
            self.user_feature_embeddings.append(e_)
        self.user_feature_embeddings = torch.nn.ModuleList(self.user_feature_embeddings)

        self.item_feature_embeddings = []
        for f_num in self.item_feature_num_list:
            e_ = torch.nn.Linear(
                in_features=f_num,
                out_features=self.embedding_dim,
                bias=False
            )
            # torch.nn.init.normal_(e_.weight, std=0.1)
            self.item_feature_embeddings.append(e_)
        self.item_feature_embeddings = torch.nn.ModuleList(self.item_feature_embeddings)


        # model

        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

        self.gcn_layer_number = config['gcn_layer_number']

        # self.build_features()

    # def build_features(self):
    #     # warm_user_ids = set(self.gcn_dataset.warm_user_ids)
    #     # warm_item_ids = set(self.gcn_dataset.warm_item_ids)
    #
    #     u_fs_list = [[] for _ in range(self.num_u_feature)]
    #
    #     for u_id in range(self.gcn_dataset.n_user):
    #         if u_id in self.gcn_dataset.user_dict:
    #             cur_f = self.gcn_dataset.user_dict[u_id]
    #             for f_idx in range(self.num_u_feature):
    #                 u_fs_list[f_idx].append(cur_f[f_idx].float())
    #         else:
    #             for f_idx in range(self.num_u_feature):
    #                 u_fs_list[f_idx].append(torch.zeros((1, self.user_feature_num_list[f_idx]), dtype=torch.float))
    #
    #     u_f_mask = []
    #     for f_idx in range(self.num_u_feature):
    #         cur_fs = torch.cat(u_fs_list[f_idx], dim=0)
    #         if self.use_cuda:
    #             cur_fs = cur_fs.cuda()
    #         u_f_mask.append(cur_fs)
    #
    #     self.u_f_mask = u_f_mask
    #
    #     # item
    #
    #     i_fs_list = [[] for _ in range(self.num_i_feature)]
    #
    #     for i_id in range(self.gcn_dataset.m_item):
    #         if i_id in self.gcn_dataset.item_dict:
    #             cur_f = self.gcn_dataset.item_dict[i_id]
    #             for f_idx in range(self.num_i_feature):
    #                 i_fs_list[f_idx].append(cur_f[f_idx].float())
    #         else:
    #             for f_idx in range(self.num_i_feature):
    #                 i_fs_list[f_idx].append(torch.zeros((1, self.item_feature_num_list[f_idx]), dtype=torch.float))
    #
    #     i_f_mask = []
    #     for f_idx in range(self.num_i_feature):
    #         cur_fs = torch.cat(i_fs_list[f_idx], dim=0)
    #         if self.use_cuda:
    #             cur_fs = cur_fs.cuda()
    #         i_f_mask.append(cur_fs)
    #
    #     self.i_f_mask = i_f_mask


    def computer_gcn(self, A_hat):

        users_emb = self.user_gcn_emb.weight
        items_emb = self.item_gcn_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
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




    def forward(self, user_features, item_features, user_ids, item_ids, A_hat, training=True):
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

        all_users_gcn_emb, all_items_gcn_emb = self.computer_gcn(A_hat)

        user_gcn_emb = all_users_gcn_emb[user_ids]
        item_gcn_emb = all_items_gcn_emb[item_ids]

        split_user_features = []
        for f_idx in range(self.num_u_feature):
            cur_f_emb_ = self.user_embeddings[f_idx](user_features[f_idx])
            split_user_features.append(cur_f_emb_)

        emb_all_user_features = torch.cat(split_user_features, dim=1)

        split_item_features = []
        for f_idx in range(self.num_i_feature):
            cur_f_emb_ = self.item_embeddings[f_idx](item_features[f_idx])
            split_item_features.append(cur_f_emb_)

        emb_all_item_features = torch.cat(split_user_features, dim=1)



        # do combination


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
        self.local_lr = config['local_lr']
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        # TODO: change name
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']

        self.A_hat_train = gcn_dataset.getSparseGraph() # cache=False
        self.gcn_dataset = gcn_dataset


    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_y, support_features, support_pair_id, query_features, query_pair_id, num_local_update):
        supp_user_features = support_features[0]
        supp_item_features = support_features[1]

        query_user_feature = query_features[0]
        query_item_feature = query_features[1]

        support_uid, support_iid = support_pair_id[:, 0], support_pair_id[:, 1]
        query_uid, query_iid = query_pair_id[:, 0], query_pair_id[:, 1]

        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(supp_user_features, supp_item_features, support_uid, support_iid, self.A_hat_train)
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
        query_set_y_pred = self.model(query_user_feature, query_item_feature, query_uid, query_iid, self.A_hat_train)
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

    def global_update(self, batch_data, num_local_update):

        (s_pair_batch, s_featur_batch, s_y_batch,
         q_pair_batch, q_featur_batch, q_y_batch) = batch_data

        batch_sz = len(s_pair_batch)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                s_pair_batch[i] = s_pair_batch[i].cuda()
                # s_featur_batch[i] = s_featur_batch[i]
                s_y_batch[i] = s_y_batch[i].cuda()

                q_pair_batch[i] = q_pair_batch[i].cuda()
                # q_featur_batch[i] = q_featur_batch[i].cuda()
                q_y_batch[i] = q_y_batch[i].cuda()

        # support query
        for i in range(batch_sz):
            query_set_y_pred = self.forward(s_y_batch[i], s_featur_batch[i], s_pair_batch[i],
                                            q_featur_batch[i], q_pair_batch[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, q_y_batch[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
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
