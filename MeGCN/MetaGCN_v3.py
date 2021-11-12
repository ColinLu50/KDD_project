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
        # self.fc1_in_dim = config['embedding_dim'] * 10
        # self.fc2_in_dim = config['first_fc_hidden_dim']
        # self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']
        self.gcn_dataset = gcn_dataset



        # self.item_emb = ItemEmb(config, gcn_dataset.m_item)
        # self.user_emb = UserEmb(config, gcn_dataset.n_user)
        # GCN feature embedding
        self.num_users = gcn_dataset.n_user
        self.num_items = gcn_dataset.m_item
        self.num_user_features = gcn_dataset.num_user_feature
        self.num_item_features = gcn_dataset.num_item_feature


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
        self.user_emb = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim
        )
        self.item_emb = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim
        )
        self.features_emb = torch.nn.Embedding(
            num_embeddings=(self.num_user_features + self.num_item_features),
            embedding_dim=self.embedding_dim
        )



        # with torch.no_grad():
        #     warm_uids = set(self.gcn_dataset.warm_user_ids)
        #
        #     for u_id in range(self.num_users):
        #         if u_id not in warm_uids:
        #             self.user_gcn_emb.weight[u_id, :] = 0
        #
        #     warm_iids = set(self.gcn_dataset.warm_item_ids)
        #
        #     for i_id in range(self.num_items):
        #         if i_id not in warm_iids:
        #             self.item_gcn_emb.weight[i_id, :] = 0


        # self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        # self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        # self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

        self.gcn_layer_number = config['gcn_layer_number']

        # self.layer_weight = torch.nn.Linear(
        #         in_features=self.embedding_dim,
        #         out_features=self.embedding_dim,
        #         bias=False
        #     )

        # self.layer_weight = torch.nn.Linear(
        #     in_features=self.embedding_dim,
        #     out_features=self.embedding_dim,
        #     bias=False
        # )
        N = self.num_users + self.num_items + self.num_user_features + self.num_item_features
        self.layer_weight = torch.nn.Parameter(data=torch.Tensor(N, 32), requires_grad=True)
        self.layer_weight.data.normal_(std=0.5)





    def computer_gcn(self, A_hat):

        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        features_emb = self.features_emb.weight
        all_emb = torch.cat([users_emb, items_emb, features_emb])
        embs = [all_emb]

        g = A_hat

        for layer in range(self.gcn_layer_number):
            all_emb = torch.mul(self.layer_weight, all_emb)
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users_gcn, items_gcn, features_gcn = torch.split(light_out, [self.num_users, self.num_items,
                                                       self.num_user_features + self.num_item_features])

        return users_gcn, items_gcn, features_gcn




        # with torch.no_grad():
        #     for uid in warm_user_ids:
        #         self.embedding_all_user[uid] =
        #
        #     for iid in warm_item_ids:
        #         self.embedding_all_item[iid] = torch.nn.Parameter(torch.ones(self.embedding_dim).cuda())

    def get_init_emb(self, user_ids, item_ids):
        return self.user_emb(user_ids), self.item_emb(item_ids)



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

        all_users_gcn_emb, all_items_gcn_emb, all_features_gcn_emb = self.computer_gcn(A_hat)

        user_gcn_emb = all_users_gcn_emb[user_ids]
        item_gcn_emb = all_items_gcn_emb[item_ids]

        # user_gcn_emb = self.layer_weight(user_gcn_emb)
        # item_gcn_emb = self.layer_weight(item_gcn_emb)

        inner_product = torch.mul(user_gcn_emb, item_gcn_emb)

        gcn_outputs = torch.sum(inner_product, dim=1)


        return gcn_outputs




class MetaGCN(torch.nn.Module):
    def __init__(self, config, gcn_dataset):
        super(MetaGCN, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = GCN_Estimator(config, gcn_dataset)
        self.local_lr = config['local_lr']
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        # TODO: change name
        # self.local_update_target_weight_name = ['user_gcn_emb.weight', 'item_gcn_emb.weight',
        #                                         'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
        #                                         'linear_out.weight', 'linear_out.bias']

        # self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
        #                                         'linear_out.weight', 'linear_out.bias']

        self.local_update_target_weight_name = ['layer_weight.weight']

        self.A_hat_train = gcn_dataset.getSparseGraph() # cache=False
        self.gcn_dataset = gcn_dataset


    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_y, support_pair_id, query_pair_id, num_local_update):
        # supp_user_features = [f[0] for f in support_features]
        # supp_item_features = [f[1] for f in support_features]
        #
        # query_user_feature = [f[0] for f in query_features]
        # query_item_feature = [f[1] for f in query_features]

        support_uid, support_iid = support_pair_id[:, 0], support_pair_id[:, 1]
        query_uid, query_iid = query_pair_id[:, 0], query_pair_id[:, 1]

        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_uid, support_iid, self.A_hat_train)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
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

    def inference(self, support_set_y, support_features, support_pair_id, query_features, query_pair_id, num_local_update):
        # build new hat
        new_A_hat = self.gcn_dataset.getNewSparseGraph(torch.cat([support_pair_id, query_pair_id]))

        supp_user_features = [f[0] for f in support_features]
        supp_item_features = [f[1] for f in support_features]

        query_user_feature = [f[0] for f in query_features]
        query_item_feature = [f[1] for f in query_features]

        support_uid, support_iid = support_pair_id[:, 0], support_pair_id[:, 1]
        query_uid, query_iid = query_pair_id[:, 0], query_pair_id[:, 1]

        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(supp_user_features, supp_item_features, support_uid, support_iid,
                                            new_A_hat)
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
        query_set_y_pred = self.model(query_user_feature, query_item_feature, query_uid, query_iid, new_A_hat)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def get_weights_decay_loss(self, query_pair_id):
        # support_uid, support_iid = support_pair_id[:, 0], support_pair_id[:, 1]
        query_uid, query_iid = query_pair_id[:, 0], query_pair_id[:, 1]
        user_emb0, item_emb0 = self.model.get_init_emb(query_uid, query_iid)

        weight_decay_loss = (1/2)*(user_emb0.norm(2).pow(2) + item_emb0.norm(2).pow(2))/float(query_iid.shape[0])

        return weight_decay_loss



    def global_update(self, batch_data, num_local_update):

        (s_pair_batch, s_y_batch,
         q_pair_batch, q_y_batch) = batch_data

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


        for i in range(batch_sz):
            query_set_y_pred = self.forward(s_y_batch[i], s_pair_batch[i],
                                            q_pair_batch[i], num_local_update)
            # TODO: get weight decay

            loss_q = F.mse_loss(query_set_y_pred, q_y_batch[i]) + self.get_weights_decay_loss(q_pair_batch[i])
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return losses_q

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
