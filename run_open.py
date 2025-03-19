# -*- coding: utf-8 -*-
import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models_CVAE import *
import numpy as np
import pickle
import os
import argparse
import sys
import datetime
import signal
from PIL import Image
from loss import CondDiscCtn, loss_function, ContrastiveLoss, SoftmaxCrossEntropyLoss
from utils import *
from torch.distributions import Normal, kl_divergence
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def get_args():
    parser = argparse.ArgumentParser(description='Train for open-set reg')
    parser.add_argument('--phase', default="test", type=str, choices='train or test',
                        help="train or test")
    parser.add_argument('--ROC', default=False, choices='calculate roc curve and AUROC')
    parser.add_argument('--config', default="mstar_10.yaml", type=str, help="experiment settings")
    parser.add_argument('--epochs', default=500, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=20, type=int, help="Batch size")
    parser.add_argument('--num_classes', default=8, type=int, help="Number of known classes in dataset")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--save_path', default="./saved_models/Aexp2_without_contrastive/opentrain", type=str,
                        help="Path to save the ensemble weights")
    parser.add_argument('--load_path', default="./saved_models/Aexp2_without_contrastive", type=str,
                        help="Path to save the ensemble weights")
    parser.add_argument('--margin', default=3, type=float, help="contrastive margin")
    parser.add_argument('--roc_saved_path', default="./saved_roc/sample_unk2/Ours.pkl", type=str,
                        help="roc_saved_path")
    parser.set_defaults(argument=True)

    return parser.parse_args()


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.console_log_path = os.path.join(path, 'console_output',
                                             '{}.txt'.format(
                                                 datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.exists(os.path.dirname(self.console_log_path)):
            os.makedirs(os.path.dirname(self.console_log_path))
        self.log = open(self.console_log_path, 'a')
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C.')
        self.log.close()

        # Remove logfile
        # os.remove(self.console_log_path)
        print('Save log file at:', self.console_log_path)
        sys.exit(0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def Gen_cond_vec(label_lst, num_classes):
    cond_nmatch_batch = []
    cond_match_batch = []
    cond_matrix = torch.full((num_classes, num_classes), 0.0)
    cond_matrix.diagonal().fill_(1.0)  # a row is a condvec
    index_lst = torch.arange(0, num_classes).cuda()
    for i in range(len(label_lst)):
        label = label_lst[i]
        # matching conditional vector generating
        cond_match_batch.append(cond_matrix[label, :])
        # non-matching conditional vector generating
        ind_nmatch_lst = index_lst[index_lst != label]  # remove matching label from full label list
        ind_nmatch = torch.randint(0, len(ind_nmatch_lst), (1,))
        cond_nmatch_batch.append(ind_nmatch_lst[ind_nmatch])
    cond_nmatch_batch = torch.cat(cond_nmatch_batch, dim=0).cuda()
    # cond_match_batch = torch.stack(cond_match_batch, dim=0).cuda()
    # cond_match_batch = cond_match_batch.repeat_interleave(num_classes - 1, dim=0)
    # return cond_match_batch, cond_nmatch_batch
    return cond_nmatch_batch


def epoch_train(classencoder, decoder, condaug, unk_detector, trainloader, optimizer_glb, epoch,
                args):
    classencoder.eval()
    condaug.train()
    unk_detector.train()
    decoder.eval()

    num_classes = args.num_classes

    losses = AverageMeter('Loss', ':6.2f')
    celosses = AverageMeter('CELoss', ':6.2f')
    contralosses = AverageMeter('ContraLoss', ':6.2f')
    accs = AverageMeter('Acc', ':3.1f')
    progress = ProgressMeter(
        len(trainloader),
        [losses, celosses, contralosses, accs],
        prefix="Epoch: [{}]".format(epoch))
    criteron = ContrastiveLoss(margin=args.margin)

    label_vector = torch.arange(0, num_classes).cuda()
    label_onehot = F.one_hot(label_vector, num_classes)
    for iteration, data in enumerate(trainloader):
        # detector training
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        cond_dists = torch.zeros(images.size()[0], num_classes).cuda()
        _, class_mulogvar = classencoder(label_vector)
        for j in range(num_classes):
            class_mulogvar_etd = class_mulogvar[j, :].repeat(images.size()[0], 1)
            class_mu, class_logvar = torch.split(class_mulogvar_etd, args.latent_dim, dim=1)
            label_onehot_etd = label_onehot[j, :].repeat(images.size()[0], 1)
            z = reparameterize(class_mu, class_logvar)
            recons = decoder(z=z, y=label_onehot_etd.float())
            dist, embedd_ori = condaug(images, recons)
            cond_dists[:, j] = dist.squeeze()
        y_logits, w_logits, prob = unk_detector(images, cond_dists)

        dist_m = cond_dists[torch.arange(cond_dists.size(0)), labels]
        labels_nm = Gen_cond_vec(labels, num_classes)
        dist_nm = cond_dists[torch.arange(cond_dists.size(0)), labels_nm]
        dist = torch.cat((dist_m, dist_nm), dim=0)

        contrast_labels = torch.cat(
            (torch.zeros(dist_m.size()[0]), torch.ones(dist_nm.size()[0]))
            , dim=0)
        contrast_loss = criteron(dist, contrast_labels.to(dist.device))
        CE_loss = 0.5 * F.cross_entropy(y_logits, labels) + 0.5 * F.cross_entropy(w_logits, labels)
        loss = args.alpha * contrast_loss + args.beta * CE_loss

        optimizer_glb.zero_grad()
        loss.backward()
        optimizer_glb.step()

        y_pre = torch.argmax(prob, dim=1)
        acc = torch.mean((y_pre == labels).float()) * 100
        losses.update(loss.item(), dist_nm.size(0))
        celosses.update(CE_loss.item(), dist_nm.size(0))
        contralosses.update(contrast_loss.item(), dist_nm.size(0))
        accs.update(acc.item(), dist_nm.size(0))
        if iteration % args.print_freq == 0 or iteration == (len(trainloader) - 1):
            progress.display(iteration)
    return


def epoch_openval(inputencoder, classencoder, decoder, condaug, unk_detector, testopen_loader, args):
    inputencoder.eval()
    classencoder.eval()
    decoder.eval()
    condaug.eval()
    unk_detector.eval()

    num_classes = args.num_classes

    open_conddiscaccs = AverageMeter('Open_conddisc Acc', ':3.1f')

    progress = ProgressMeter(
        len(testopen_loader),
        [open_conddiscaccs],
        prefix='Test: ')

    label_vector = torch.arange(0, num_classes).cuda()
    _, class_mulogvar = classencoder(label_vector)
    label_onehot = F.one_hot(label_vector, num_classes)
    labels_lst = []
    pre_lst = []
    unkprob_lst = []
    for i, data in enumerate(testopen_loader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        labels[labels > num_classes] = num_classes
        labels_lst.append(labels.cpu().detach())
        # images_etd = images.repeat_interleave(num_classes, dim=0)
        cond_dists = torch.zeros(images.size()[0], num_classes).cuda()
        for j in range(num_classes):
            class_mulogvar_etd = class_mulogvar[j, :].repeat(images.size()[0], 1)
            class_mu, class_logvar = torch.split(class_mulogvar_etd, args.latent_dim, dim=1)
            label_onehot_etd = label_onehot[j, :].repeat(images.size()[0], 1)
            z = reparameterize(class_mu, class_logvar)
            recons = decoder(z=z, y=label_onehot_etd.float())
            dist, embedd_ori = condaug(images, recons)
            cond_dists[:, j] = dist.squeeze()
        _, _, full_prob = unk_detector(images, cond_dists)
        unkprob_lst.append(1 - torch.max(full_prob.cpu().detach(), dim=1).values)
        KL_chunks = KL_dist(inputencoder, classencoder, images, args)
        y_pres = torch.zeros_like(labels)
        for j in range(len(labels)):
            max_prob = torch.max(full_prob[j, :])
            if max_prob < args.thresh:
                y_pres[j] = num_classes
            else:
                KL_chunk = KL_chunks[j]
                _, KLypre = torch.min(KL_chunk, dim=0)
                y_pres[j] = KLypre
        pre_lst.append(y_pres.detach())
        acc = torch.mean((y_pres == labels).float()) * 100
        open_conddiscaccs.update(acc.item(), labels.size(0))

        if i % args.print_freq == 0 or (i == (len(testopen_loader) - 1)):
            progress.display(i)
    if args.ROC:
        unkprob = np.array(torch.cat(unkprob_lst))
        labels = np.array(torch.cat(labels_lst))
        unkprob_k = unkprob[labels < num_classes]
        unkprob_u = unkprob[labels == num_classes]
        print("The AUROC is ", calc_auroc(unkprob_u, unkprob_k))
        roc_data = [unkprob_u, unkprob_k]
        with open(args.roc_saved_path, 'wb') as file:
            pickle.dump(roc_data, file)
        # save_roc(unkprob_u, unkprob_k, saved_path=args.roc_saved_path)
        # return
    print(' * Open_conddisc Acc@1 {KL.avg:.3f}'.format(KL=open_conddiscaccs))

    return open_conddiscaccs.avg, labels_lst, pre_lst


def KL_dist(inputencoder, classencoder, images, args):
    label_matrix = torch.arange(0, args.num_classes).cuda()
    _, cls_mulogvar = classencoder(label_matrix)
    clsexp_mulogvar = cls_mulogvar.repeat(images.size()[0], 1)
    cls_mu, cls_logvar = torch.split(clsexp_mulogvar, args.latent_dim, dim=1)

    images_etd = images.repeat_interleave(args.num_classes, dim=0)
    _, input_mulogvar = inputencoder(images_etd)
    input_mu, input_logvar = torch.split(input_mulogvar, args.latent_dim, dim=1)

    kld_loss = -0.5 * torch.sum(1 + (input_logvar - cls_logvar)
                                - torch.pow(cls_mu - input_mu, 2) / torch.exp(cls_logvar)
                                - torch.exp(input_logvar) / torch.exp(cls_logvar), dim=1)
    KL_chunks = torch.chunk(kld_loss, chunks=images.size(0), dim=0)
    return KL_chunks


def metrics(labels_lst, pre_lst):
    label = np.array(torch.concatenate(labels_lst).cpu())
    result = np.array(torch.concatenate(pre_lst).cpu())

    recall = recall_score(label, result, average='macro')
    print("宏召回率:", recall)

    precision = precision_score(label, result, average='macro')
    print("宏精确率:", precision)

    f1 = f1_score(label, result, average='macro')
    print("宏F1值:", f1)

    cm = confusion_matrix(label, result)
    print(cm)


def main():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    arg_params = get_args()
    with open("./open_config/{}".format(arg_params.config)) as file:
        configs = yaml.safe_load(file)

    args = merge_params(arg_params, configs)
    sys.stdout = Logger(args.save_path)
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    momentum = args.momentum
    means = args.means
    stds = args.stds
    print(args)

    num_classes = args.num_classes
    print("Num classes " + str(num_classes))

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    print(transform_train)
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    print(transform_test)

    root = args.dataset_dir

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(root, "train_8"),
                                                transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, pin_memory=True, drop_last=False)

    testopenset = torchvision.datasets.ImageFolder(root=os.path.join(root, "val_8"),
                                                   transform=transform_test)
    testopen_loader = torch.utils.data.DataLoader(testopenset, batch_size=batch_size,
                                                  shuffle=False, pin_memory=True, drop_last=False)

    inputencoder = InputEncoder(latent_dim=args.latent_dim)
    inputencoder = inputencoder.cuda()
    decoder = Decoder(latent_dim=args.latent_dim, num_classes=args.num_classes)
    decoder = decoder.cuda()
    classencoder = ClassEncoder(latent_dim=args.latent_dim, img_size=128, num_classes=args.num_classes)
    classencoder = classencoder.cuda()
    condaug = CondReFeaAug(margin=args.margin)
    condaug = condaug.cuda()
    unk_detector = Open_corrector(num_classes=args.num_classes)
    unk_detector = unk_detector.cuda()

    # load model
    checkpoint_close = torch.load(os.path.join(args.load_path, 'closetrain/checkpoints/best.pth'))
    inputencoder.load_state_dict(checkpoint_close['inputencoder_state_dict'])
    classencoder.load_state_dict(checkpoint_close['classencoder_state_dict'])
    decoder.load_state_dict(checkpoint_close['decoder_state_dict'])
    # checkpoint_open = torch.load(os.path.join(args.load_path, 'opentrain/checkpoints/best.pth'))
    # unk_detector.load_state_dict(checkpoint_open['unk_detector_state_dict'])
    # condaug.load_state_dict(checkpoint_open['condaug_state_dict'])
    # decoder.load_state_dict(checkpoint_open['decoder_state_dict'])

    optimizer_glb = optim.SGD([
        {'params': unk_detector.parameters()},
        {'params': condaug.parameters()}
    ], lr=lr, momentum=momentum)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    if args.phase == 'test':
        checkpoint_close = torch.load(os.path.join(args.load_path, 'closetrain/checkpoints/best.pth'))
        inputencoder.load_state_dict(checkpoint_close['inputencoder_state_dict'])
        classencoder.load_state_dict(checkpoint_close['classencoder_state_dict'])
        decoder.load_state_dict(checkpoint_close['decoder_state_dict'])
        checkpoint_open = torch.load(os.path.join(args.load_path, 'opentrain/checkpoints/best.pth'))
        unk_detector.load_state_dict(checkpoint_open['unk_detector_state_dict'])
        condaug.load_state_dict(checkpoint_open['condaug_state_dict'])
        _, label_lst, pre_lst = epoch_openval(inputencoder, classencoder, decoder, condaug, unk_detector,
                                              testopen_loader,  args)
        if not args.ROC:
            metrics(label_lst, pre_lst)
        return

    best_acc = 0.
    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_train(classencoder, decoder, condaug, unk_detector, trainloader, optimizer_glb, epoch, args)
        test_acc, label_lst, pre_lst = epoch_openval(inputencoder, classencoder, decoder, condaug, unk_detector,
                                                     testopen_loader, args)
        metrics(label_lst, pre_lst)
        torch.save({'epoch': epoch,
                    'unk_detector_state_dict': unk_detector.state_dict(),
                    'condaug_state_dict': condaug.state_dict(),
                    'decoder_state_dict': decoder.state_dict()},
                   os.path.join(args.save_path, 'checkpoints', 'latest.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch,
                        'unk_detector_state_dict': unk_detector.state_dict(),
                        'condaug_state_dict': condaug.state_dict(),
                        'decoder_state_dict': decoder.state_dict()},
                       os.path.join(args.save_path, 'checkpoints', 'best.pth'))
        print("Best Open-set Test Acc: {:.3f}".format(best_acc))


if __name__ == "__main__":
    main()
