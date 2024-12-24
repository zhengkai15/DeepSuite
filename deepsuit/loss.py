import torch
import torch.nn as nn
from loguru import logger

class weighted_mae(nn.Module):
    def __init__(self,config):
        super(weighted_mae, self).__init__()
        self.weight = config["loss"]["weighted_mae_weight"]

    def forward(self, preds, gts):
        loss = torch.mean(torch.abs(preds - gts) * torch.pow(self.weight, gts))
        return loss

class TverskyLossPlusMSESplit(nn.Module):
    def __init__(self, ALPHA=1.0, BETA=1.0):
        super(TverskyLossPlusMSESplit, self).__init__()
        self.alpha, self.beta = ALPHA, BETA
        self.mse = nn.MSELoss()
    def forward(self, _inputs, _targets, smooth=1-9) :
        mse_loss = self.mse(_inputs, _targets)
        inputs = _inputs. sigmoid().view(-1)
        targets = _targets. sigmoid().view(-1)

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        tversky = (TP + smooth)/(TP + self.alpha * FP + self.beta * FN + smooth)
        tversky_loss = 1 - tversky
        return tversky_loss + mse_loss

class TverskyLossPlusWEIGHTMAESplit(nn.Module):
    def __init__(self, ALPHA=1.0, BETA=1.0):
        super(TverskyLossPlusWEIGHTMAESplit, self).__init__()
        self.alpha, self.beta = ALPHA, BETA
        self._weighted_mae = weighted_mae()
    def forward(self, _inputs, _targets, smooth=1-9) :
        mse_loss = self._weighted_mae(_inputs, _targets)
        inputs = _inputs. sigmoid().view(-1)
        targets = _targets. sigmoid().view(-1)

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        tversky = (TP + smooth)/(TP + self.alpha * FP + self.beta * FN + smooth)
        tversky_loss = 1 - tversky
        return tversky_loss + mse_loss

# rank.6
class WeightedMSELoss(nn.Module):
    def __init__(self, config):
        super(WeightedMSELoss, self).__init__()
        self.base = config["loss"]["base"]
        self.normalize = config["loss"]["normalize"]
        self.targets_div = config["loss"]["targets_div"]

    def forward(self, inputs, targets, fuxi_pred=None, log1p_target=False):
        # print(targets.max(), targets.min())
        # Normalize targets to [0, 1] range based on known max value
        scaled_targets = targets / self.targets_div
        # Calculate exponential weights
        weights = torch.exp(self.base * scaled_targets)
        # Normalize weights if specified
        if self.normalize:
            weights /= weights.sum()
        # Calculate the weighted MSE
        loss = (weights * (inputs - targets) ** 2).mean()
        return loss
    

# zhangqi.rank.10
class LabelBasedWeightedRegressionLoss(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.use_mae = config["loss"]["use_mae"]
        self.nonzero_mean = config["loss"]["nonzero_mean"]
        if self.use_mae:
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            self.loss_fn = nn.MSELoss(reduction='none')
        self.weights = config["loss"]["weights"]

    def forward(self, pred, label, fuxi_pred=None, log1p_target=False):
        """
        Computes the weighted mean squared error loss based on the specified conditions.

        Args:
            pred (torch.Tensor): Predictions of shape [bs, 3, 57, 81].
            label (torch.Tensor): Ground truth labels of shape [bs, 3, 57, 81].

        Returns:
            torch.Tensor: The computed weighted MSE loss.
        """
        assert pred.shape == label.shape, "Prediction and label must have the same shape"

        w_c1, w_c2, w_c3, w_c4, w_c5 = self.weights

        error = self.loss_fn(pred, label)

        weights = torch.ones_like(label)
        weights = weights.to(dtype=error.dtype, device=error.device)

        if log1p_target:
            # fuxi_pred = torch.expm1(fuxi_pred)
            label = torch.expm1(label)

        non_zero_elements = (label != 0.0).sum()
        if non_zero_elements == 0:
            return error.mean()

        condition1 = label <= 1.
        condition2 = (label >= 1.) & (label < 10.)
        condition3 = (label >= 10.) & (label < 30.)
        condition4 = (label >= 30.) & (label < 50.)
        condition5 = (label >= 50.)

        weights = torch.where(
            condition1,
            torch.tensor(w_c1, dtype=error.dtype, device=error.device),
            weights)
        weights = torch.where(
            condition2,
            torch.tensor(w_c2, dtype=error.dtype, device=error.device),
            weights)
        weights = torch.where(
            condition3,
            torch.tensor(w_c3, dtype=error.dtype, device=error.device),
            weights)
        weights = torch.where(
            condition4,
            torch.tensor(w_c4, dtype=error.dtype, device=error.device),
            weights)
        weights = torch.where(
            condition5,
            torch.tensor(w_c5, dtype=error.dtype, device=error.device),
            weights)

        weighted_error = error * weights

        if self.nonzero_mean:
            loss = weighted_error.sum() / non_zero_elements
        else:
            loss = weighted_error.mean()
        return loss



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