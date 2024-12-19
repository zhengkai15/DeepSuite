# -*-coding:utf-8 -*-
import os
import random
import yaml
import numpy as np
import torch
import argparse
from loguru import logger

def get_color_codes():
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_RESET = '\033[49m'  # 重置背景颜色
    return BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_RESET


def parse_args():
    """Parse command line arguments for model training configuration."""
    parser = argparse.ArgumentParser(
        description="Parse configuration arguments."
    )
    # config
    parser.add_argument(
        '--config',
        type=str,
        default='/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/sais-race.2024/demo_review_1h_log/conf/config.ymal',
    )

    # mode
    parser.add_argument(
        '--mode',
        type=str,
        default='train'
    )

    # Training hyperparameters
    parser.add_argument(
        '--lr.value',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    return parser.parse_args()
    

def print_config(config):
    """
    打印配置文件中的所有变量，按表格格式输出。
    
    :param config: 配置字典或包含配置的对象
    """
    # 遍历config字典中的所有项并添加到表格
    for key, value in config.items():
        # 只记录符合条件的对象
        if "__" not in key:
            logger.info(f"{key}:   {value}" )
            
            
def update_config_from_args(config, args):
    """
    Update a nested dictionary (config) with flat keys (updates).
    Example:
        config = {'a': {'b': 1}}
        updates = {'a.b': 2}
        Result: {'a': {'b': 2}}
    """
    for key, value in vars(args).items():
        logger.info(f"***args: {key}   {value}***")
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        # print(keys[-1], d[keys[-1]])
    logger.info(20*"*")

    print_config(config)
    return config


def set_random_seed(seed: int):
    try:
        from lightning import seed_everything
        seed_everything(seed, workers=True)
    except:
        # Python 随机数生成器
        random.seed(seed)
        
        # NumPy 随机数生成器
        np.random.seed(seed)
        
        # PyTorch 随机数生成器
        torch.manual_seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # GPU 随机数生成器
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 为了确保结果可重复
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def load_yaml(file_path):
    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



