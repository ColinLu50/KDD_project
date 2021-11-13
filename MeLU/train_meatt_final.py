import os
import sys
import torch
import pickle

from MeAtt_final import MeAtt
from MeAtt_config import config
from data_generation import generate
from evaluation_meatt import evaluation_
import random
random.seed(1)




def training(model_, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        model_.cuda()

    best_ever = -1

    training_set_size = len(total_dataset)
    model_.train()
    for epoch in range(num_epoch):
        random.shuffle(total_dataset)
        epoch_loss = []

        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            batch_loss = model_.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])
            epoch_loss.append(batch_loss)

        print(f'Epoch {epoch} Loss: {torch.stack(epoch_loss).mean(0)}')
        sys.stdout.flush()

        if model_save:
            cur_v = evaluation_(model_, master_path, 'tmp', test_state="user_and_item_cold_state")
            if cur_v > best_ever:
                best_ever = cur_v
                # print('Save')
                print('Better value:', best_ever, 'Save!')
                torch.save(model_, model_filename)

print('===============================')
for k in config:
    print(k, ':', config[k])

if __name__ == "__main__":
    master_path= "/home/workspace/big_data/KDD_projects_data/ml1m"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about 22GB of your hard disk space.
        generate(master_path)

    # training model.
    melu = MeAtt(config)
    model_filename = "{}/MeAtt1.pkl".format(master_path)


    # Load training dataset.
    training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []

    for idx in range(training_set_size):
        supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))

    total_dataset = list(
        zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    # training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
    training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=20, model_save=True, model_filename=model_filename)

    torch.save(melu, model_filename)

    # test_model = torch.load(model_filename)

    evaluation_(melu, master_path, 'MeAtt1_final')