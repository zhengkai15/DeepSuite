from .utils import logger
import torch.nn as nn

def get_loss(config):
    loss_name = config["loss"]["name"]
    """Get loss function based on loss name.
    
    Args:
        loss_name (str): Name of the loss function.
        
    Returns:
        loss_func: The loss function object.
    """
    if loss_name == 'weighted_mae':
        logger.info(f"loss_name:{loss_name}")
        loss_func = weighted_mae(config)
        logger.info(f'weight:{loss_func.weight}')
    # rank.6
    elif loss_name == "WeightedMSELoss":
        logger.info(f"loss_name:{loss_name}")
        loss_func = WeightedMSELoss(config)
    # rank.10
    elif loss_name == 'LabelBasedWeightedRegressionLoss':
        logger.info(f"loss_name:{loss_name}")
        loss_func = LabelBasedWeightedRegressionLoss(config)

    elif loss_name == 'mse':
        logger.info(f"loss_name:{loss_name}")
        loss_func = nn.MSELoss()

    elif loss_name == 'TverskyLossPlusMSESplit':
        logger.info(f"loss_name:{loss_name}")
        loss_func = TverskyLossPlusMSESplit()

    elif loss_name == 'TverskyLossPlusWEIGHTMAESplit':
        logger.info(f"loss_name:{loss_name}")
        loss_func = TverskyLossPlusWEIGHTMAESplit()

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
        
    return loss_func