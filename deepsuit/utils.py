import os
import sys
from loguru import logger
from datetime import datetime, timedelta

# 获取当前子模块的日志记录器
# logger = logging.getLogger(__name__)  # 名称自动是 'module.submodule1'
from loguru import logger

def time_me(func):
    '''
    @summary: cal the time of the fucntion
    @param : None
    @return: return the res of the func
    '''
    def wrapper(*args,**kw):
        start_time = datetime.now()
        res = func(*args, **kw)
        over_time = datetime.now()
        logger.info('current Function: {0} :run time is :{1} seconds'.format(func.__name__ , (over_time - start_time).total_seconds()))
        return res
    return wrapper


def create_logger(write_to_file=True):
    """
    创建日志记录器，支持控制是否写入文件。

    Args:
        write_to_file (bool): 是否将日志写入文件。默认 True。
    
    Returns:
        logger: 配置完成的 loguru 日志记录器。
    """
    # 移除默认配置
    logger.remove()
    
    # 配置控制台日志
    logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}")
    
    # 配置文件日志（可选）
    if write_to_file:
        now = (datetime.now() + timedelta(hours=8)).strftime("%Y%m%d_%H-%M-%S")
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        log_file = f'{script_dir}/../logs/log_statistic_{now}.log'
        
        # 创建日志文件目录
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 添加日志文件记录器
        logger.add(log_file, rotation="10 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}")
    
    return logger

logger = create_logger(write_to_file=False)