import torch
import torch.nn as nn
from loguru import logger
import math


class weighted_mae(nn.Module):
    def __init__(self,config):
        super(weighted_mae, self).__init__()
        self.weight = config["loss"]["weighted_mae_weight"]

    def forward(self, preds, gts):
        assert preds.shape == gts.shape, "Prediction and label must have the same shape"
        loss = torch.mean(torch.abs(preds - gts) * torch.pow(self.weight, gts))
        return loss

class TverskyLossPlusMSESplit(nn.Module):
    def __init__(self, ALPHA=1.0, BETA=1.0):
        super(TverskyLossPlusMSESplit, self).__init__()
        self.alpha, self.beta = ALPHA, BETA
        self.mse = nn.MSELoss()
    def forward(self, _inputs, _targets, smooth=1-9) :
        assert _inputs.shape == _targets.shape, "Prediction and label must have the same shape"
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
        assert _inputs.shape == _targets.shape, "Prediction and label must have the same shape"
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
        assert inputs.shape == targets.shape, "Prediction and label must have the same shape"
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
        self.log1p_target=config["data"]["log_transform"]

    def forward(self, pred, label, fuxi_pred=None):
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

        if self.log1p_target:
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


def reverse_to_dbz(x, config):
    thresholds_mapping_param = config["loss"]["thresholds_mapping_param"]
    x = x * thresholds_mapping_param
    return x
    
'''
class LabelBasedWeightedRegressionLoss_yx_nowcasting(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.use_mae = config["loss"]["use_mae"]
        self.nonzero_mean = config["loss"]["nonzero_mean"]
        if self.use_mae:
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            self.loss_fn = nn.MSELoss(reduction='none')
        self.config = config

    def forward(self, pred, label):
        """
        Computes the weighted mean squared error loss based on the specified conditions.

        Args:
            pred (torch.Tensor): Predictions of shape [bs, 3, 57, 81].
            label (torch.Tensor): Ground truth labels of shape [bs, 3, 57, 81].

        Returns:
            torch.Tensor: The computed weighted MSE loss.
        """
        assert pred.shape == label.shape, "Prediction and label must have the same shape"

        w_c1, w_c2, w_c3, w_c4, w_c5, w_c6, w_c7 = self.config["loss"]["weights"]

        error = self.loss_fn(pred, label)

        weights = torch.ones_like(label)
        weights = weights.to(dtype=error.dtype, device=error.device)

        label = reverse_to_dbz(label, self.config)
        pred = reverse_to_dbz(pred, self.config)

        non_zero_elements = (label != 0.0).sum()
        if non_zero_elements == 0:
            return error.mean()

        condition1 = label <= 1.
        condition2 = (label >= 1.) & (label < 10.)
        condition3 = (label >= 10.) & (label < 20.)
        condition4 = (label >= 20.) & (label < 30.)
        condition5 = (label >= 30.) & (label < 40.)
        condition6 = (label >= 40.) & (label < 50.)
        condition7 = (label >= 50.)

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
        weights = torch.where(
            condition6,
            torch.tensor(w_c6, dtype=error.dtype, device=error.device),
            weights)
        
        weights = torch.where(
            condition7,
            torch.tensor(w_c7, dtype=error.dtype, device=error.device),
            weights)
        
        weighted_error = error * weights

        if self.nonzero_mean:
            loss = weighted_error.sum() / non_zero_elements
        else:
            loss = weighted_error.mean()
        return loss
'''
    
class LabelBasedWeightedRegressionLoss_yx_nowcasting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_mae = config["loss"]["use_mae"]
        self.nonzero_mean = config["loss"]["nonzero_mean"]
        self.loss_fn = nn.L1Loss(reduction='none') if self.use_mae else nn.MSELoss(reduction='none')
        self.config = config
        
    def gen_weights(self, label):
        thresholds = self.config["loss"]["thresholds"]
        weights = self.config["loss"]["weights"]
        # weight的个数需要大于thresholds的个数
        assert len(thresholds)+1 == len(weights), "thresholds length equals to weight lenght plus 1"
    
        # thresholds个阈值
        conditions = []
        for i in range(len(thresholds)):
            if i == 0:
                conditions.append(label <= thresholds[i])
            else:
                conditions.append((label > thresholds[i-1]) & (label <= thresholds[i]))
        conditions.append(label > thresholds[-1])
        return conditions
        
    def forward(self, pred, label):
        assert pred.shape == label.shape, "Prediction and label must have the same shape"
        
        # ae
        error = self.loss_fn(pred, label)
        
        # 数据处理
        label = reverse_to_dbz(label, self.config)
        pred = reverse_to_dbz(pred, self.config)
        non_zero_elements = (label != 0.0).sum()
        
        if non_zero_elements == 0:
            return error.mean()
        
        # 生成condition
        conditions = self.gen_weights(label)
        
        # 生成weight
        stacked_conditions = torch.stack(conditions)
        weights = torch.tensor(self.config["loss"]["weights"], dtype=error.dtype, device=error.device)
        weights = weights.view(len(weights), *([1] * (len(stacked_conditions.shape) - 1))).expand_as(stacked_conditions)
        weights = torch.where(stacked_conditions, weights, torch.ones_like(label, dtype=error.dtype, device=error.device))
        weighted_error = error * weights
        loss = weighted_error.sum() / non_zero_elements if self.nonzero_mean else weighted_error.mean()
        return loss


import torch.nn.functional as F
class mae_plus_mse(nn.Module):
    def __init__(self, config):
        super(mae_plus_mse, self).__init__()

    def forward(self, pred, obs):
        assert pred.shape == obs.shape, "Prediction and label must have the same shape"
        l1_loss = F.l1_loss(pred, obs, reduction='mean')
        mse_loss = F.mse_loss(pred, obs, reduction='mean')
        return l1_loss + mse_loss
    
    
import numpy as np
class BalancedL1Loss(torch.nn.Module):
    def __init__(
            self,
            config,
            gamma=0.5,
            momentum=0.95,
    ):
        super(BalancedL1Loss, self).__init__()
        bins = config["loss"]["thresholds"]
        logger.info(f"thresholds:{bins}")
        bins = list(zip(bins[:-1], bins[1:]))
        self.bins = np.array(bins, dtype=np.float32)
        self.gamma = gamma
        self.momentum = momentum
        self.num_bin = len(self.bins)
        counts = torch.full((self.num_bin,), 1e4, dtype=torch.float32)
        # counts = torch.full((self.num_bin,), 1e1, dtype=torch.float32)
        self.register_buffer("counts", counts, persistent=False)
    @torch.no_grad()
    def compute_weight(self, targets):
        weights = torch.ones_like(targets)
        masks = []
        for i in range(self.num_bin):
            th1, th2 = self.bins[i]
            mask = (targets >= th1) & (targets < th2)
            masks.append(mask)
            self.counts[i] = (
                    self.momentum * self.counts[i] + (1 - self.momentum) * mask.sum()
            )
        counts = self.counts
        for i in range(self.num_bin):
            freq = max(1, counts[i]) / counts.sum()
            wi = (1 / freq.item()) ** self.gamma
            weights[masks[i]] = wi
        return weights
    def forward(self, outputs, targets, weight=None):
        assert outputs.shape == targets.shape, "Prediction and label must have the same shape"
        if weight is None:
            weight = self.compute_weight(targets)
        else:
            weight = self.compute_weight(targets) * weight
        loss = F.l1_loss(outputs, targets, reduction="none")
        loss = torch.sum(loss * weight) / weight.sum()
        return loss
    
import numpy as np
class BalancedL1Loss_fix(torch.nn.Module):
    def __init__(
            self,
            config,
            gamma=0.5,
            momentum=0.95,
    ):
        super(BalancedL1Loss_fix, self).__init__()
        logger.info(f'thresholds:{config["loss"]["thresholds"]}')
        bins = config["loss"]["thresholds"]
        thresholds_mapping_param = config["loss"]["thresholds_mapping_param"]
        if config["data"]["log_transform"] and config["data"]["min_max_transform"]:
            raise ValueError("log_transform and min_max_transform cannot both be true")
        elif config["data"]["log_transform"]:
            bins = [math.log(x + 1) for x in bins]
            logger.info(f"log_transform fixed thresholds:{bins}")
        elif config["data"]["min_max_transform"]:
            bins = [i/thresholds_mapping_param for i in bins]
            logger.info(f"min_max_transform fixed thresholds:{bins}")
        else:
            logger.info(f"no transform to thresholds:{bins}")
        bins = list(zip(bins[:-1], bins[1:]))
        self.bins = np.array(bins, dtype=np.float32)
        self.gamma = gamma
        self.momentum = momentum
        self.num_bin = len(self.bins)
        counts = torch.full((self.num_bin,), 1e4, dtype=torch.float32)
        # counts = torch.full((self.num_bin,), 1e1, dtype=torch.float32)
        self.register_buffer("counts", counts, persistent=False)
    @torch.no_grad()
    def compute_weight(self, targets):
        weights = torch.ones_like(targets)
        masks = []
        for i in range(self.num_bin):
            th1, th2 = self.bins[i]
            mask = (targets >= th1) & (targets < th2)
            masks.append(mask)
            self.counts[i] = (
                    self.momentum * self.counts[i] + (1 - self.momentum) * mask.sum()
            )
        counts = self.counts
        for i in range(self.num_bin):
            freq = max(1, counts[i]) / counts.sum()
            wi = (1 / freq.item()) ** self.gamma
            weights[masks[i]] = wi
        return weights
    def forward(self, outputs, targets, weight=None):
        assert outputs.shape == targets.shape, "Prediction and label must have the same shape"
        if weight is None:
            weight = self.compute_weight(targets)
        else:
            weight = self.compute_weight(targets) * weight
        loss = F.l1_loss(outputs, targets, reduction="none")
        loss = torch.sum(loss * weight) / weight.sum()
        return loss
    
class mse_loss(torch.nn.Module):
    def __init__(self, config):
        super(mse_loss, self).__init__()
    
    def forward(self, outputs, targets):
        assert outputs.shape == targets.shape, "Prediction and label must have the same shape"
        loss = F.mse_loss(outputs, targets, reduction='mean')
        return loss

class BMAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BMAELoss, self).__init__()
        self.reduction = 'mean'
        self.config = config
        self.thresholds = self.config["loss"]["thresholds"]
        self.weights = self.config["loss"]["weights"]
        assert len(thresholds)+1 == len(weight), "thresholds length equals to weight lenght plus 1"
        self.b_dict = {self.thresholds[i]: self.weights[i+1] for i in range(len(self.thresholds))}
        
    def get_weight(self, y):
        w = torch.ones_like(y) * self.weight[0]
        for k in sorted(self.b_dict.keys()):
            w[y >= k] = self.b_dict[k]
        return w 
        
    def forward(self, outputs, targets):
        assert outputs.shape == targets.shape, "Prediction and label must have the same shape"
        w = torch.sqrt(self.get_weight(targets))
        return F.l1_loss(w * outputs, w * targets, reduction=self.reduction)

def get_loss(config):
    loss_name = config["loss"]["name"]
    """Get loss function based on loss name.
    
    Args:
        loss_name (str): Name of the loss function.
        
    Returns:
        loss_func: The loss function object.
    """
    logger.info(f"loss_name:{loss_name}")
    
    if loss_name == 'weighted_mae':
        loss_func = weighted_mae(config)
        logger.info(f'weight:{loss_func.weight}')
    # rank.6
    elif loss_name == "WeightedMSELoss":
        loss_func = WeightedMSELoss(config)
    # rank.10
    elif loss_name == 'LabelBasedWeightedRegressionLoss':
        loss_func = LabelBasedWeightedRegressionLoss(config)

    elif loss_name == 'mse':
        loss_func = nn.MSELoss()

    elif loss_name == 'TverskyLossPlusMSESplit':
        loss_func = TverskyLossPlusMSESplit()

    elif loss_name == 'TverskyLossPlusWEIGHTMAESplit':
        loss_func = TverskyLossPlusWEIGHTMAESplit()
    # yx_nowcasting
    elif loss_name == 'LabelBasedWeightedRegressionLoss_yx_nowcasting':
        loss_func = LabelBasedWeightedRegressionLoss_yx_nowcasting(config) 
        
    elif loss_name ==  "mae_plus_mse":
        # loss_func = lambda pred, obs: (F.l1_loss(pred, obs, reduction='mean') + F.mse_loss(pred, obs, reduction='mean'))
        loss_func = mae_plus_mse(config)
        
    elif loss_name == "balanced_l1_loss":
        loss_func = BalancedL1Loss(config)
    
    elif loss_name == "balanced_l1_loss_fix":
        loss_func = BalancedL1Loss_fix(config)
        
    elif loss_name == "mse_loss":
        loss_func = mse_loss(config)
        
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
        
    return loss_func