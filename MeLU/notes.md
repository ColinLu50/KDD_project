

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

MeGCN 平行增加 32 user and item embedding.
Task warm_state, NDCG1:  0.8877, nDCG3:  0.8905 NDCG5:  0.9043, nDCG10:  0.9611
Task user_cold_state, NDCG1:  0.8787, nDCG3:  0.8826 NDCG5:  0.8937, nDCG10:  0.9576
Task item_cold_state, NDCG1:  0.7593, nDCG3:  0.7805 NDCG5:  0.8104, nDCG10:  0.9198
Task user_and_item_cold_state, NDCG1:  0.7626, nDCG3:  0.7790 NDCG5:  0.8086, nDCG10:  0.9193


