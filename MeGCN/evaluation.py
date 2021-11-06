import os
import torch
import pickle
from sklearn.metrics import ndcg_score
import numpy as np
from tqdm import tqdm


from options import config, states

# torch.autograd.set_detect_anomaly(True)

def evaluation_(megcn, master_path, log_name):
    # if not os.path.exists("{}/scores/".format(master_path)):
    #     os.mkdir("{}/scores/".format(master_path))
    if config['use_cuda']:
        megcn.cuda()
    megcn.eval()

    all_info = pickle.load(open("{}/all_info.pkl".format(master_path), "rb"))
    max_uid, max_mid, warm_uids, warm_iids, state_size = all_info

    result_str = ''

    for target_state in states:
        ndcg1_list = []
        ndcg3_list = []
        ndcg5_list = []
        ndcg10_list = []
        dataset_size = state_size[target_state]
        for idx in tqdm(list(range(dataset_size))):

            support_ys = pickle.load(open("{}/{}/supp_y_{}.pkl".format(master_path, target_state, idx), "rb")).cuda()
            support_pair_id = pickle.load(open("{}/{}/supp_pairs_{}.pkl".format(master_path, target_state, idx), "rb")).cuda()
            query_pair_id = pickle.load(open("{}/{}/query_pairs_{}.pkl".format(master_path, target_state, idx), "rb")).cuda()
            query_ys = pickle.load(open("{}/{}/query_y_{}.pkl".format(master_path, target_state, idx), "rb"))


            # item_ids = []
            # with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, target_state, j), "r") as f:
            #     for line in f.readlines():
            #         item_id = line.strip().split()[1]
            #         item_ids.append(item_id)

            query_y_pred = megcn.forward(support_ys, support_pair_id, query_pair_id, config['inner']) #config['inner']
            query_y_pred = query_y_pred.view(1, -1).cpu().detach().numpy()
            ndcg1 = ndcg_score(query_ys.view(1, -1).numpy(), query_y_pred, k=1)
            ndcg1_list.append(ndcg1)
            ndcg3 = ndcg_score(query_ys.view(1, -1).numpy(), query_y_pred, k=3)
            ndcg3_list.append(ndcg3)

            ndcg5 = ndcg_score(query_ys.view(1, -1).numpy(), query_y_pred, k=5)
            ndcg5_list.append(ndcg5)

            ndcg10 = ndcg_score(query_ys.view(1, -1).numpy(), query_y_pred, k=10)
            ndcg10_list.append(ndcg10)


        print(f'Task {target_state}, NDCG1: {np.mean(ndcg1_list) : .4f}, nDCG3: {np.mean(ndcg3_list) : .4f} NDCG5: {np.mean(ndcg5_list) : .4f}, nDCG10: {np.mean(ndcg10_list) : .4f}')
        result_str += f'\nTask {target_state}, NDCG1: {np.mean(ndcg1_list) : .4f}, nDCG3: {np.mean(ndcg3_list) : .4f} NDCG5: {np.mean(ndcg5_list) : .4f}, nDCG10: {np.mean(ndcg10_list) : .4f}'


    file_path = os.path.join(master_path, 'out', log_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, 'w') as f:
        f.write(result_str + '\n')


if __name__ == "__main__":
    from MetaGCN import MetaGCN
    from gcn_dataloader import GCNDataLoader
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m_test"

    # training model.
    ml_dataset = GCNDataLoader(master_path)
    # melu = MetaGCN(config, ml_dataset)
    model_filename = "{}/test_MetaGCN.pkl".format(master_path)
    if not os.path.exists(model_filename):
        raise Exception(f'Model not exist in {master_path}')
    else:
        megcn = torch.load(model_filename)
        # melu.load_state_dict(trained_state_dict)

    evaluation_(megcn, master_path, 'megcn_2_simple')