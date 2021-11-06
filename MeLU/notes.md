

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

Task warm_state, NDCG1:  0.8815, nDCG3:  0.8857 NDCG5:  0.8999, nDCG10:  0.9594
Task user_cold_state, NDCG1:  0.8716, nDCG3:  0.8765 NDCG5:  0.8890, nDCG10:  0.9556
Task item_cold_state, NDCG1:  0.7856, nDCG3:  0.8069 NDCG5:  0.8330, nDCG10:  0.9290
Task user_and_item_cold_state, NDCG1:  0.7778, nDCG3:  0.8024 NDCG5:  0.8286, nDCG10:  0.9276