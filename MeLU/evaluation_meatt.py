import os
import sys
import torch
import pickle
from sklearn.metrics import ndcg_score
import numpy as np
from tqdm import tqdm


from options import config, states

# torch.autograd.set_detect_anomaly(True)

def evaluation_(melu, master_path, log_name, test_state=None):
    # if not os.path.exists("{}/scores/".format(master_path)):
    #     os.mkdir("{}/scores/".format(master_path))
    if config['use_cuda']:
        melu.cuda()
    melu.eval()

    result_str = ''



    for target_state in states:

        if test_state is not None and target_state != test_state:
            continue

        ndcg1_list = []
        ndcg3_list = []
        ndcg5_list = []
        ndcg10_list = []
        dataset_size = int(len(os.listdir("{}/{}".format(master_path, target_state))) / 4)
        for j in tqdm(list(range(dataset_size)), disable=True):
            support_xs = pickle.load(open("{}/{}/supp_x_{}.pkl".format(master_path, target_state, j), "rb"))
            support_ys = pickle.load(open("{}/{}/supp_y_{}.pkl".format(master_path, target_state, j), "rb"))
            query_xs = pickle.load(open("{}/{}/query_x_{}.pkl".format(master_path, target_state, j), "rb"))
            query_ys = pickle.load(open("{}/{}/query_y_{}.pkl".format(master_path, target_state, j), "rb"))

            # item_ids = []
            # with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, target_state, j), "r") as f:
            #     for line in f.readlines():
            #         item_id = line.strip().split()[1]
            #         item_ids.append(item_id)

            query_y_pred = melu.forward(support_xs.cuda(), support_ys.cuda(), query_xs.cuda(),
                                         config['inner']) #config['inner']
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

    print('=' * 30)
    print(result_str)

    sys.stdout.flush()

    if test_state is not None:
        return np.mean(ndcg1_list)

    result_log_folder = os.path.join(master_path, 'out')
    if not os.path.exists(result_log_folder):
        os.makedirs(result_log_folder)

    with open(os.path.join(result_log_folder, log_name), 'w') as f:
        f.write(result_str + '\n')


if __name__ == "__main__":
    from MeLU import MeLU

    master_path = "/home/workspace/big_data/KDD_projects_data/ml1m"

    # training model.
    # melu = MeLU(config)
    model_filename = "{}/MeLU5_test.pkl".format(master_path)
    if not os.path.exists(model_filename):
        raise Exception(f'Model not exist in {master_path}')
    else:
        melu = torch.load(model_filename)
        # melu.load_state_dict(trained_state_dict)


    evaluation_(melu, master_path, 'melu5_test_evl')
    melu3 = MeLU(config)
    melu.load_state_dict(torch.load("{}/MeLU5_test_state.pkl".format(master_path)))
    melu.store_parameters()
    evaluation_(melu3, master_path, 'melu5_test_evl')