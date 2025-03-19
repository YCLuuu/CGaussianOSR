import torch
import torch.nn.functional as F
from torch import nn


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def CondDiscCtn(d_m, d_nonm, w_m=None, w_nonm=None):
    d_label_m = torch.zeros((d_m.size(0), 1)).to(d_m.device)
    d_label_nonm = torch.ones((d_nonm.size(0), 1)).to(d_nonm.device)
    cond_discriminator_accuracy = 0.5 * (
            binary_accuracy(d_m, d_label_m) + binary_accuracy(d_nonm, d_label_nonm))

    conddisc_loss = 0.5 * F.binary_cross_entropy(d_m, d_label_m) + 0.5 * F.binary_cross_entropy(d_nonm, d_label_nonm)

    return conddisc_loss, cond_discriminator_accuracy


def loss_function(recons, input, input_mu, input_logvar, class_mu, class_logvar, kld_weight) -> dict:
    # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(recons, input)

    ## KL(q(z|x,c)||p(z|c))
    kld_loss = torch.mean(-0.5 * torch.sum(1 + (input_logvar - class_logvar)
                                           - torch.pow(class_mu - input_mu, 2) / torch.exp(class_logvar)
                                           - torch.exp(input_logvar) / torch.exp(class_logvar), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

def loss_vae(recons, input, input_mu, input_logvar, kld_weight) -> dict:
    # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(recons, input)

    ## KL(q(z|x,c)||p(z|c))
    kld_loss = torch.mean(-0.5 * torch.sum(1 + input_logvar - input_mu ** 2 - input_logvar.exp(), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, target):
        # euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            target * torch.pow(torch.max(torch.tensor(0.0), self.margin - euclidean_distance), 2) +
            (1 - target) * torch.pow(euclidean_distance, 2))
        return loss_contrastive


def SoftmaxCrossEntropyLoss(softmax_output, targets):
    loss = F.nll_loss(softmax_output, targets)

    return loss
