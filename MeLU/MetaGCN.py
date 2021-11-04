import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

# from embeddings import item, user


class ItemEmb(torch.nn.Module):
    def __init__(self, config, m_item):
        super(ItemEmb, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_director = config['num_director']
        self.num_actor = config['num_actor']
        self.embedding_dim = config['embedding_dim']

        # self.num_item = m_item
        # self.embedding_item_content = torch.nn.Embedding(
        #     num_embeddings=self.num_item,
        #     embedding_dim=self.embedding_dim
        # )


        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, item_id,  vars=None):
        # item_content_emb = self.embedding_item_content(item_id)
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)

# class ItemEmb(torch.nn.Module):
#     def __init__(self, config, m_item):
#         super(ItemEmb, self).__init__()
#         self.num_rate = config['num_rate']
#         self.num_genre = config['num_genre']
#         self.num_director = config['num_director']
#         self.num_actor = config['num_actor']
#         self.embedding_dim = config['embedding_dim']
#
#         self.num_item = m_item
#         self.embedding_item_content = torch.nn.Embedding(
#             num_embeddings=self.num_item,
#             embedding_dim=self.embedding_dim
#         )
#
#
#         self.embedding_rate = torch.nn.Embedding(
#             num_embeddings=self.num_rate,
#             embedding_dim=self.embedding_dim
#         )
#
#         self.embedding_genre = torch.nn.Embedding(
#             num_embeddings=self.num_genre,
#             embedding_dim=self.embedding_dim
#         )
#
#         self.embedding_director = torch.nn.Embedding(
#             num_embeddings=self.num_director,
#             embedding_dim=self.embedding_dim
#         )
#
#         self.embedding_actor = torch.nn.Embedding(
#             num_embeddings=self.num_actor,
#             embedding_dim=self.embedding_dim
#         )
#
#
#     def forward(self, rate_idx, genre_idx, director_idx, actors_idx, item_id,  vars=None):
#         item_content_emb = self.embedding_item_content(item_id)
#         rate_emb = self.embedding_rate(rate_idx)
#         genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
#         director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
#         actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
#         return torch.cat((rate_emb, genre_emb, director_emb, actors_emb, item_content_emb), 1)


class UserEmb(torch.nn.Module):
    def __init__(self, config, n_user):
        super(UserEmb, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']

        # self.num_user = n_user
        # self.embedding_user_content = torch.nn.Embedding(
        #     num_embeddings=self.num_user,
        #     embedding_dim=self.embedding_dim
        # )

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx, u_id):
        # user_content_emb = self.embedding_user_content(u_id)
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)


class GCN_Estimator(torch.nn.Module):
    def __init__(self, config, gcn_dataset):
        super(GCN_Estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 10
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.user_embedding_dim = config['embedding_dim']
        self.item_embedding_dim = config['embedding_dim']

        self.item_emb = ItemEmb(config, gcn_dataset.m_item)
        self.user_emb = UserEmb(config, gcn_dataset.n_user)

        self.num_users = gcn_dataset.n_user
        self.num_items = gcn_dataset.m_item

        self.embedding_user_relation = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.user_embedding_dim).cuda()
        self.embedding_item_relation = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.item_embedding_dim).cuda()

        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

        self.gcn_layer_number = config['gcn_layer_number']
        self.graph = gcn_dataset.getSparseGraph(cache=False)


    # def forward(self, aux_info, user_ids, item_ids, training=True):
    #     rate_idx = Variable(aux_info[:, 0], requires_grad=False)
    #     genre_idx = Variable(aux_info[:, 1:26], requires_grad=False)
    #     director_idx = Variable(aux_info[:, 26:2212], requires_grad=False)
    #     actor_idx = Variable(aux_info[:, 2212:10242], requires_grad=False)
    #     gender_idx = Variable(aux_info[:, 10242], requires_grad=False)
    #     age_idx = Variable(aux_info[:, 10243], requires_grad=False)
    #     occupation_idx = Variable(aux_info[:, 10244], requires_grad=False)
    #     area_idx = Variable(aux_info[:, 10245], requires_grad=False)
    #
    #     user_ids = Variable(user_ids, requires_grad=False)
    #     item_ids = Variable(item_ids, requires_grad=False)
    #
    #     item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx, item_ids)
    #     user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx, user_ids)
    #     X = torch.cat((item_emb, user_emb), 1)
    #
    #     # # TODO: perform GCN
    #     # for _ in range(self.gcn_layer_number):
    #
    #
    #
    #
    #     X = self.fc1(X)
    #     X = F.relu(X)
    #     X = self.fc2(X)
    #     X = F.relu(X)
    #     return self.linear_out(X)

    def forward(self, aux_info, user_ids, item_ids, training=True):
        rate_idx = Variable(aux_info[:, 0], requires_grad=False)
        genre_idx = Variable(aux_info[:, 1:26], requires_grad=False)
        director_idx = Variable(aux_info[:, 26:2212], requires_grad=False)
        actor_idx = Variable(aux_info[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(aux_info[:, 10242], requires_grad=False)
        age_idx = Variable(aux_info[:, 10243], requires_grad=False)
        occupation_idx = Variable(aux_info[:, 10244], requires_grad=False)
        area_idx = Variable(aux_info[:, 10245], requires_grad=False)

        user_ids = Variable(user_ids, requires_grad=False)
        item_ids = Variable(item_ids, requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx, item_ids)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx, user_ids)

        all_users, all_items = self.computer_gcn()

        user_gcn_emb = all_users[user_ids]
        item_gcn_emb = all_items[item_ids]
        X = torch.cat((item_emb, user_emb, user_gcn_emb, item_gcn_emb), 1)

        # # TODO: perform GCN
        # for _ in range(self.gcn_layer_number):




        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        return self.linear_out(X)

    def computer_gcn(self):

        users_emb = self.embedding_user_relation.weight
        items_emb = self.embedding_item_relation.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        g = self.graph

        for layer in range(self.gcn_layer_number):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


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

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, support_uid, support_iid, query_uid, query_iid, num_local_update):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x, support_uid, support_iid)
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
        query_set_y_pred = self.model(query_set_x, query_uid, query_iid)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update_MAML(self, batch_data, num_local_update):

        (support_set_xs, support_set_ys, query_set_xs, query_set_ys,
         support_uids, support_iids, query_uids, query_iids) = batch_data

        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()

                support_uids[i] = support_uids[i].cuda()
                support_iids[i] = support_iids[i].cuda()
                query_uids[i] = query_uids[i].cuda()
                query_iids[i] = query_iids[i].cuda()

        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i],
                                            support_uids[i], support_iids[i], query_uids[i], query_iids[i],
                                            num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
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
