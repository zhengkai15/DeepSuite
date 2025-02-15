import os
import random
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset, DataLoader, Subset

# 定义闭包 worker_init_fn
def create_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    return worker_init_fn
    
    
def get_dataloader(config, mydataset=None):
    worker_init_fn = create_worker_init_fn(config["seed"]["random_seed"])
    my_data_train = mydataset(flag='train',config=config)
    train_subset_indices =  np.random.choice(my_data_train.__len__(), config["train"]["iter_num"]*config["train"]["batch_size"], replace=False)
    my_data_train_subsets = Subset(my_data_train, train_subset_indices)
    train_loader = DataLoader(my_data_train_subsets, batch_size=config["train"]["batch_size"], num_workers=config["train"]["num_workers"], shuffle=config["train"]["shuffle"], worker_init_fn=worker_init_fn)

    my_data_train_eval = mydataset(flag='train',config=config) 
    train_eval_subset_indices =  np.random.choice(my_data_train_eval.__len__(), config["infer"]["iter_num"]*config["infer"]["batch_size"], replace=False)
    my_data_train_eval_subsets = Subset(my_data_train_eval, train_eval_subset_indices)
    train_loader_eval = DataLoader(my_data_train_eval_subsets, batch_size=config["infer"]["batch_size"], num_workers=config["infer"]["num_workers"], shuffle=config["infer"]["shuffle"], worker_init_fn=worker_init_fn)

    my_data_valid = mydataset(flag='valid',config=config)
    valid_subset_indices =  np.random.choice(my_data_valid.__len__(), config["infer"]["iter_num"]*config["infer"]["batch_size"], replace=False)
    my_data_valid_subsets = Subset(my_data_valid, valid_subset_indices)
    valid_loader = DataLoader(my_data_valid_subsets, batch_size=config["infer"]["batch_size"], num_workers=config["infer"]["num_workers"], shuffle=config["infer"]["shuffle"], worker_init_fn=worker_init_fn)

    my_data_test = mydataset(flag='test',config=config)
    test_subset_indices =  np.random.choice(my_data_test.__len__(), config["infer"]["iter_num"]*config["infer"]["batch_size"], replace=False)
    my_data_test_subsets = Subset(my_data_test, test_subset_indices)
    test_loader = DataLoader(my_data_test_subsets, batch_size=config["infer"]["batch_size"], num_workers=config["infer"]["num_workers"], shuffle=config["infer"]["shuffle"], worker_init_fn=worker_init_fn)

    return train_loader, train_loader_eval, valid_loader, test_loader
    # my_data_test = mydataset(flag='test',config=config)
    # test_loader = DataLoader(my_data_test, batch_size=1, num_workers=1, shuffle=False, worker_init_fn=worker_init_fn)

    # return None, None, None, test_loader