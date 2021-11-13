import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from torch import nn

from embeddings import item, user


class attention_estimator(torch.nn.Module):
    def __init__(self, config):
        super(attention_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        # self.fc1_in_dim = config['embedding_dim'] * 2
        # self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.item_emb = item(config)
        self.user_emb = user(config)
        N = self.embedding_dim * 8
        self.fc1 = torch.nn.Linear(N, N)
        self.fc2 = torch.nn.Linear(N, self.embedding_dim * 4)

        self.fc3 = torch.nn.Linear(N, self.embedding_dim * 4)


        self.linear_out = torch.nn.Linear(N, 1)

    def forward(self, x, training = True):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        x = self.fc1(x)
        layer1_act = F.relu(x)

        user_attention = F.softmax(self.fc2(layer1_act), dim=1)
        item_attention = F.softmax(self.fc3(layer1_act), dim=1)

        user_emb = user_attention * user_emb
        item_emb = item_attention * item_emb

        _y = self.linear_out(torch.cat((item_emb, user_emb), 1))

        norm_y = F.softmax(_y, dim=0) * 4 + 1

        return norm_y

class transformer_estimator(torch.nn.Module):
    def __init__(self, config):
        super(transformer_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.item_emb = item(config)
        self.user_emb = user(config)
        N = self.embedding_dim * 4 # 32 * 4 = 128
        self.multihead_num = 8
        self.multihead_dim = N // self.multihead_num
        # self.transformer = torch.nn.Transfomer()

        self.factor = self.multihead_dim ** (1/2)

        query_user_list = []
        for _ in range(self.multihead_num):
            query_user_list.append(
                torch.nn.Linear(N, self.multihead_dim,  bias=False)
            )
        self.query_user_list = nn.ModuleList(query_user_list)

        self.key_user_list = []
        for _ in range(self.multihead_num):
            self.key_user_list.append(
                torch.nn.Linear(N, self.multihead_dim, bias=False)
            )
        self.key_user_list = nn.ModuleList(self.key_user_list)

        self.value_user_list = []
        for _ in range(self.multihead_num):
            self.value_user_list.append(
                torch.nn.Linear(N, self.multihead_dim, bias=False)
            )
        self.value_user_list = nn.ModuleList(self.value_user_list)

        self.query_item_list = []
        for _ in range(self.multihead_num):
            self.query_item_list.append(
                torch.nn.Linear(N, self.multihead_dim, bias=False)
            )
        self.query_item_list = nn.ModuleList(self.query_item_list)

        self.key_item_list = []
        for _ in range(self.multihead_num):
            self.key_item_list.append(
                torch.nn.Linear(N, self.multihead_dim, bias=False)
            )
        self.key_item_list = nn.ModuleList(self.key_item_list)


        self.value_item_list = []
        for _ in range(self.multihead_num):
            self.value_item_list.append(
                torch.nn.Linear(N, self.multihead_dim, bias=False)
            )
        self.value_item_list = nn.ModuleList(self.value_item_list)

        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

        # self.linear_out = torch.nn.Linear(N, 1)


    def attention_qkv(self, queries, keys, values):
        w_ = F.softmax(queries.matmul(keys.T) / self.factor, dim=1)
        return w_.matmul(values)




    def forward(self, x, training = True):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx[0].unsqueeze(0), age_idx[0].unsqueeze(0), occupation_idx[0].unsqueeze(0), area_idx[0].unsqueeze(0))

        # user
        user_multi_head = []
        items_multi_head = []
        for i in range(self.multihead_num):

            q_user_i = self.query_user_list[i](user_emb)
            k_user_i = self.key_user_list[i](user_emb)
            v_user_i = self.value_user_list[i](user_emb)

            q_item_i = self.query_item_list[i](item_emb)
            k_item_i = self.key_item_list[i](item_emb)
            v_item_i = self.value_item_list[i](item_emb)

            # user to items
            v_u_i = self.attention_qkv(q_user_i, k_item_i, v_item_i)

            # user self-attention
            v_u_u = self.attention_qkv(q_user_i, k_user_i, v_user_i)

            v_u = v_u_u + v_u_i
            user_multi_head.append(v_u) # (1, # items)

            # items to items
            v_i_i = self.attention_qkv(q_item_i, k_item_i, v_item_i)
            v_i_u = self.attention_qkv(q_item_i, k_user_i, v_user_i)
            v_i = v_i_i + v_i_u

            items_multi_head.append(v_i)

        # for i in range(self.multihead_num):
        #
        #     q_user_i = self.query_user_list[i](user_emb)
        #     k_user_i = self.key_user_list[i](user_emb)
        #     v_user_i = self.value_user_list[i](user_emb)
        #
        #     q_item_i = self.query_item_list[i](item_emb)
        #     k_item_i = self.key_item_list[i](item_emb)
        #     v_item_i = self.value_item_list[i](item_emb)
        #
        #     # user to items
        #     v_u_i = self.attention_qkv(q_user_i, k_item_i, v_item_i)
        #
        #     # user self-attention
        #     # v_u_u = self.attention_qkv(q_user_i, k_user_i, v_user_i)
        #
        #     v_u = v_u_i
        #     user_multi_head.append(v_u) # (1, # items)
        #
        #     # items to items
        #     # v_i_i = self.attention_qkv(q_item_i, k_item_i, v_item_i)
        #     v_i_u = self.attention_qkv(q_item_i, k_user_i, v_user_i)
        #     v_i = v_i_u
        #
        #     items_multi_head.append(v_i)

        user_attention = torch.cat(user_multi_head, dim=1)
        items_attention = torch.cat(items_multi_head, dim=1)

        users_attetion = user_attention.repeat((x.shape[0], 1))

        X = torch.cat([users_attetion, items_attention], dim=1)


        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        _y = self.linear_out(X)

        return _y

        # norm_y = F.softmax(_y, dim=0) * 4 + 1

        # norm_y = (_y - _y.mean()) / _y.std() * 2.5 + 2.5
        #
        # return norm_y


class MeAtt(torch.nn.Module):
    def __init__(self, config):
        super(MeAtt, self).__init__()
        self.use_cuda = config['use_cuda']


        # self.model = attention_estimator(config)
        self.model = transformer_estimator(config)



        self.local_lr = config['local_lr']
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        # self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
        #                                         'fc3.weight', 'fc3.bias', 'linear_out.weight', 'linear_out.bias']



        self.local_update_target_weight_name = []
        for w_name in list(self.model.state_dict().keys()):
            if 'emb' in w_name:
                continue
            self.local_update_target_weight_name.append(w_name)
        # ['item_emb.embedding_rate.weight', 'item_emb.embedding_genre.weight', 'item_emb.embedding_director.weight',
        #  'item_emb.embedding_actor.weight', 'user_emb.embedding_gender.weight', 'user_emb.embedding_age.weight',
        #  'user_emb.embedding_occupation.weight', 'user_emb.embedding_area.weight', 'query_user_list.0.weight',
        #  'query_user_list.1.weight', 'query_user_list.2.weight', 'query_user_list.3.weight', 'query_user_list.4.weight',
        #  'query_user_list.5.weight', 'query_user_list.6.weight', 'query_user_list.7.weight', 'key_user_list.0.weight',
        #  'key_user_list.1.weight', 'key_user_list.2.weight', 'key_user_list.3.weight', 'key_user_list.4.weight',
        #  'key_user_list.5.weight', 'key_user_list.6.weight', 'key_user_list.7.weight', 'value_user_list.0.weight',
        #  'value_user_list.1.weight', 'value_user_list.2.weight', 'value_user_list.3.weight', 'value_user_list.4.weight',
        #  'value_user_list.5.weight', 'value_user_list.6.weight', 'value_user_list.7.weight', 'query_item_list.0.weight',
        #  'query_item_list.1.weight', 'query_item_list.2.weight', 'query_item_list.3.weight', 'query_item_list.4.weight',
        #  'query_item_list.5.weight', 'query_item_list.6.weight', 'query_item_list.7.weight', 'key_item_list.0.weight',
        #  'key_item_list.1.weight', 'key_item_list.2.weight', 'key_item_list.3.weight', 'key_item_list.4.weight',
        #  'key_item_list.5.weight', 'key_item_list.6.weight', 'key_item_list.7.weight', 'value_item_list.0.weight',
        #  'value_item_list.1.weight', 'value_item_list.2.weight', 'value_item_list.3.weight', 'value_item_list.4.weight',
        #  'value_item_list.5.weight', 'value_item_list.6.weight', 'value_item_list.7.weight', 'linear_out.weight',
        #  'linear_out.bias']

        print(self.local_update_target_weight_name)

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
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
        query_set_y_pred = self.model(query_set_x)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
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
