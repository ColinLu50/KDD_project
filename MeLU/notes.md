# MeLU-1
Task warm_state: MAE: 0.7857814431190491 NDCG1:  0.8813, nDCG3:  0.8860 NDCG5:  0.9009, nDCG10:  0.9595
Task user_cold_state: MAE: 0.7947486639022827 NDCG1:  0.8755, nDCG3:  0.8790 NDCG5:  0.8888, nDCG10:  0.9561
Task item_cold_state: MAE: 0.9341222643852234 NDCG1:  0.7758, nDCG3:  0.7986 NDCG5:  0.8269, nDCG10:  0.9261
Task user_and_item_cold_state: MAE: 0.9251037240028381 NDCG1:  0.7689, nDCG3:  0.7938 NDCG5:  0.8232, nDCG10:  0.9251


# Attetion
## v1
Use MLP_att
cat all user and item feature embedding, convert to user atten and item atten
attention nomalized by (Ndim^0.5)

## v2
remove nomailized by factor, just use softmax


# Transfomer

{num_rate : 6
num_genre : 25
num_director : 2186
num_actor : 8030
embedding_dim : 32
first_fc_hidden_dim : 64
second_fc_hidden_dim : 64
num_gender : 2
num_age : 7
num_occupation : 21
num_zipcode : 3402
use_cuda : True
inner : 5
lr : 5e-05
local_lr : 5e-06
batch_size : 32
num_epoch : 20}

Task warm_state: MAE: 0.8187016844749451 NDCG1:  0.8559, nDCG3:  0.8623 NDCG5:  0.8778, nDCG10:  0.9502
Task user_cold_state: MAE: 0.8149060606956482 NDCG1:  0.8511, nDCG3:  0.8593 NDCG5:  0.8729, nDCG10:  0.9489
Task item_cold_state: MAE: 0.9334436655044556 NDCG1:  0.7968, nDCG3:  0.8079 NDCG5:  0.8332, nDCG10:  0.9299
Task user_and_item_cold_state: MAE: 0.921103298664093 NDCG1:  0.7892, nDCG3:  0.8035 NDCG5:  0.8288, nDCG10:  0.9283

Task warm_state: MAE: 0.7649665474891663 NDCG1:  0.8725, nDCG3:  0.8811 NDCG5:  0.8966, nDCG10:  0.9576
Task user_cold_state: MAE: 0.8078774213790894 NDCG1:  0.8641, nDCG3:  0.8710 NDCG5:  0.8848, nDCG10:  0.9532
Task item_cold_state: MAE: 0.9595338106155396 NDCG1:  0.7767, nDCG3:  0.8058 NDCG5:  0.8310, nDCG10:  0.9278
Task user_and_item_cold_state: MAE: 0.9569819569587708 NDCG1:  0.7660, nDCG3:  0.7976 NDCG5:  0.8242, nDCG10:  0.9253

# Optimized Transfomer

## V1: local update decision

## V2: 
- local update all layer except emb
- inner loop hight lr 
- high inner loop 15

# Dataset

warm_state skip number 2179 / 4832
1208it [00:09, 120.87it/s]
user_cold_state skip number 540 / 1208
4769it [00:37, 127.95it/s]
item_cold_state skip number 1906 / 4769
1197it [00:09, 127.36it/s]
user_and_item_cold_state skip number 474 / 1197

每个user对应一个 ' supp_x_{user_idx}.pkl', 跳过了很多user <13, >100


Task warm_state, NDCG1:  0.7922, nDCG3:  0.8109 NDCG5:  0.8361, nDCG10:  0.9323
Task user_cold_state, NDCG1:  0.7715, nDCG3:  0.7999 NDCG5:  0.8299, nDCG10:  0.9282
Task item_cold_state, NDCG1:  0.7653, nDCG3:  0.7886 NDCG5:  0.8204, nDCG10:  0.9226
Task user_and_item_cold_state, NDCG1:  0.7790, nDCG3:  0.7925 NDCG5:  0.8222, nDCG10:  0.9244

MeGCN_V2 平行增加 32 user and item embedding.
Task warm_state, NDCG1:  0.8877, nDCG3:  0.8905 NDCG5:  0.9043, nDCG10:  0.9611
Task user_cold_state, NDCG1:  0.8787, nDCG3:  0.8826 NDCG5:  0.8937, nDCG10:  0.9576
Task item_cold_state, NDCG1:  0.7593, nDCG3:  0.7805 NDCG5:  0.8104, nDCG10:  0.9198
Task user_and_item_cold_state, NDCG1:  0.7626, nDCG3:  0.7790 NDCG5:  0.8086, nDCG10:  0.9193

MeGCN_V2 + WeightDecay + Local更新GCN emb + inference 更新Sup/Query
Task warm_state, NDCG1:  0.8840, nDCG3:  0.8907 NDCG5:  0.9039, nDCG10:  0.9608
Task user_cold_state, NDCG1:  0.8798, nDCG3:  0.8835 NDCG5:  0.8935, nDCG10:  0.9577
Task item_cold_state, NDCG1:  0.7526, nDCG3:  0.7817 NDCG5:  0.8118, nDCG10:  0.9197
Task user_and_item_cold_state, NDCG1:  0.7504, nDCG3:  0.7781 NDCG5:  0.8093, nDCG10:  0.9185


MeLU-5
Task warm_state, NDCG1:  0.8815, nDCG3:  0.8857 NDCG5:  0.8999, nDCG10:  0.9594
Task user_cold_state, NDCG1:  0.8716, nDCG3:  0.8765 NDCG5:  0.8890, nDCG10:  0.9556
Task item_cold_state, NDCG1:  0.7856, nDCG3:  0.8069 NDCG5:  0.8330, nDCG10:  0.9290
Task user_and_item_cold_state, NDCG1:  0.7778, nDCG3:  0.8024 NDCG5:  0.8286, nDCG10:  0.9276


Reptile Method:
Task warm_state, NDCG1:  0.7776, nDCG3:  0.8023 NDCG5:  0.8300, nDCG10:  0.9290
Task user_cold_state, NDCG1:  0.7831, nDCG3:  0.8075 NDCG5:  0.8331, nDCG10:  0.9309
Task item_cold_state, NDCG1:  0.7434, nDCG3:  0.7712 NDCG5:  0.8048, nDCG10:  0.9163
Task user_and_item_cold_state, NDCG1:  0.7438, nDCG3:  0.7717 NDCG5:  0.8080, nDCG10:  0.9174

Reptile LR-change MeLU-5
Task warm_state, NDCG1:  0.7788, nDCG3:  0.7998 NDCG5:  0.8271, nDCG10:  0.9283
Task user_cold_state, NDCG1:  0.7814, nDCG3:  0.8017 NDCG5:  0.8305, nDCG10:  0.9294
Task item_cold_state, NDCG1:  0.7177, nDCG3:  0.7480 NDCG5:  0.7885, nDCG10:  0.9088
Task user_and_item_cold_state, NDCG1:  0.7029, nDCG3:  0.7524 NDCG5:  0.7915, nDCG10:  0.9095




