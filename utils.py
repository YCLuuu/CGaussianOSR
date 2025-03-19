from typing import Optional, List
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import pickle


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def merge_params(arg_params, configs):
    merged_params = vars(arg_params).copy()  # Convert namespace to dictionary
    merged_params.update(configs)  # Merge dictionaries
    args = argparse.Namespace()
    for key, value in merged_params.items():
        setattr(args, key, value)
    return args


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def calc_auroc(ood_test_results, id_test_results):
    # calculate the AUROC
    scores = np.concatenate((ood_test_results, id_test_results))
    # scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    trues = np.array((([1] * len(ood_test_results)) + [0] * len(id_test_results)))
    result = roc_auc_score(trues, scores)
    return result


def save_roc(ood_test_results, id_test_results):
    scores = np.concatenate((ood_test_results, id_test_results))
    # scores = (scores-np.min(scores))/(np.max(scores)-np.min(scores))
    trues = np.array((([1] * len(ood_test_results)) + [0] * len(id_test_results)))
    fpr, tpr, _ = roc_curve(trues, scores)
    return fpr, tpr
