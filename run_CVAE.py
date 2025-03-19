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
from loss import CondDiscCtn, loss_function
from utils import *
from torch.distributions import Normal, kl_divergence
import yaml


def get_args():
    parser = argparse.ArgumentParser(description='CVAE for closed-set reg')
    parser.add_argument('--phase', default="train", type=str, choices='train or test',
                        help="train or test")
    parser.add_argument('--config', default="mstar_10.yaml", type=str, help="experiment settings")
    parser.add_argument('--epochs', default=500, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=100, type=int, help="Batch size")
    parser.add_argument('--num_classes', default=8, type=int, help="Number of known classes in dataset")
    parser.add_argument('--CE', default=True, type=bool, help="CE loss")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--save_path', default="./saved_models/Aexp1_without_CE/mstar_unk_2_s5/closetrain", type=str,
                        help="Path to save the ensemble weights")
    parser.add_argument('--load_path', default="./saved_models/Aexp1_without_CE/mstar_unk_2_s5", type=str,
                        help="Path to save the ensemble weights")
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


def draw_fig(inputencoder, classencoder, decoder, testclose_loader, args):
    inputencoder.eval()
    classencoder.eval()
    decoder.eval()

    num_classes = args.num_classes

    for i, data in enumerate(testclose_loader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        _, class_mulogvar = classencoder(labels)
        class_mu, class_logvar = torch.split(class_mulogvar, args.latent_dim, dim=1)
        z = reparameterize(class_mu, class_logvar)
        recons = decoder(z=z, y=F.one_hot(labels, num_classes).float())

        ori = np.array(images[0, 0, :, :].detach().cpu())
        ori = 255 * (ori - np.min(ori)) / (np.max(ori) - np.min(ori))
        ori = ori.astype("uint8")
        im_ori = Image.fromarray(ori, mode='L')
        im_ori.save(("./figs/ori/{}_{}.jpg").format(labels[0], i))
        recon = np.array(recons[0, 0, :, :].detach().cpu())
        recon = 255 * (recon - np.min(recon)) / (np.max(recon) - np.min(recon))
        recon = recon.astype("uint8")
        im_recons = Image.fromarray(recon, mode='L')
        im_recons.save(("./figs/rescons_m/{}_{}.jpg").format(labels[0], i))

        # recons = np.array(recon_nonm[0, 0, :, :].detach().cpu())
        # recons = 255 * (recons - np.min(recons)) / (np.max(recons) - np.min(recons))
        # recons = recons.astype("uint8")
        # im_recons = Image.fromarray(recons, mode='L')
        # im_recons.save(("./figs/rescons_nonm/{}_{}.jpg").format(labels[0], i))

    return


def epoch_train(inputencoder, classencoder, decoder, classifier, trainloader, optimizer, epoch, args):
    inputencoder.train()
    classencoder.train()
    decoder.train()
    classifier.train()

    num_classes = args.num_classes

    iteration = 0
    losses = AverageMeter('Loss', ':6.2f')
    recons_losses = AverageMeter('ReconsLoss', ':6.2f')
    kld_losses = AverageMeter('KlLoss', ':6.2f')
    CE_losses = AverageMeter('CELoss', ':6.2f')
    reg_accs = AverageMeter('Acc', ':3.1f')
    progress = ProgressMeter(
        len(trainloader),
        [losses, recons_losses, kld_losses, CE_losses, reg_accs],
        prefix="Epoch: [{}]".format(epoch))

    trainiter = iter(trainloader)
    try:
        while True:
            data = next(trainiter)
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # forward + backward + optimize
            _, input_mulogvar = inputencoder(images)
            y_logits = classifier(input_mulogvar)
            y_pre = torch.argmax(y_logits, dim=1)
            input_mu, input_logvar = torch.split(input_mulogvar, args.latent_dim, dim=1)

            z = reparameterize(input_mu, input_logvar)
            _, class_mulogvar = classencoder(labels)
            class_mu, class_logvar = torch.split(class_mulogvar, args.latent_dim, dim=1)

            recons = decoder(z=z, y=F.one_hot(labels, num_classes).float())

            losses_lst = loss_function(recons=recons, input=images, input_mu=input_mu, input_logvar=input_logvar,
                                       class_mu=class_mu, class_logvar=class_logvar, kld_weight=args.kld_weight)
            vae_loss = losses_lst['loss']
            recons_loss = losses_lst['Reconstruction_Loss']
            kld_loss = losses_lst['KLD']
            CE_loss = F.cross_entropy(y_logits, labels)
            loss = vae_loss + args.alpha * CE_loss
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = torch.mean((labels == y_pre.to(labels.device)).float()) * 100
            losses.update(loss.item(), images.size(0))
            recons_losses.update(recons_loss.item(), images.size(0))
            kld_losses.update(kld_loss.item(), images.size(0))
            CE_losses.update(CE_loss.item(), images.size(0))
            reg_accs.update(acc.item(), images.size(0))

            if iteration % args.print_freq == 0 or iteration == (len(trainloader) - 1):
                progress.display(iteration)
            iteration += 1
    except StopIteration:
        pass
    return losses.avg


def epoch_closeval(inputencoder, classencoder, classifier, testclose_loader, args):
    inputencoder.eval()
    classencoder.eval()
    classifier.eval()
    num_classes = args.num_classes

    close_KLaccs = AverageMeter('Close-set KLacc', ':3.1f')
    close_CEaccs = AverageMeter('Close-set CEacc', ':3.1f')

    progress = ProgressMeter(
        len(testclose_loader),
        [close_KLaccs, close_CEaccs],
        prefix='Test: ')

    # calculate mu and logvar for each class
    labels = torch.arange(0, args.num_classes).cuda()
    _, cls_mulogvar = classencoder(labels)
    for i, data in enumerate(testclose_loader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        clsexp_mulogvar = cls_mulogvar.repeat(images.size()[0], 1)
        cls_mu, cls_logvar = torch.split(clsexp_mulogvar, args.latent_dim, dim=1)

        _, input_mulogvar = inputencoder(images)
        y_logits = classifier(input_mulogvar)
        y_pre = torch.argmax(y_logits, dim=1)
        input_mulogvar = input_mulogvar.repeat_interleave(args.num_classes, dim=0)
        input_mu, input_logvar = torch.split(input_mulogvar, args.latent_dim, dim=1)

        acc = torch.mean((labels == y_pre.to(labels.device)).float()) * 100

        # KL loss
        kld_loss = -0.5 * torch.sum(1 + (input_logvar - cls_logvar)
                                    - torch.pow(cls_mu - input_mu, 2) / torch.exp(cls_logvar)
                                    - torch.exp(input_logvar) / torch.exp(cls_logvar), dim=1)
        # 将矩阵按行分块，每块为已知类别个行
        chunks = torch.chunk(kld_loss, chunks=images.size(0), dim=0)
        KLypres = []
        #### 遍历每个块并取最大值
        for chunk in chunks:
            _, KLypre = torch.min(chunk, dim=0)
            KLypres.append(KLypre)
        # 将结果列表转换为张量
        close_ypre = torch.stack(KLypres, dim=0)

        # losses.update(loss.item(), images.size(0))
        close_KLacc = torch.mean((labels == close_ypre.to(labels.device)).float()) * 100
        close_KLaccs.update(close_KLacc.item(), images.size(0))
        close_CEaccs.update(acc.item(), images.size(0))
        if i % args.print_freq == 0 or (i == (len(testclose_loader) - 1)):
            progress.display(i)

    print(' * Test Close-set KLAcc@1 {KL.avg:.3f}'.format(KL=close_KLaccs))

    return close_KLaccs.avg


def main():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    arg_params = get_args()
    with open("./cvae_config/{}".format(arg_params.config)) as file:
        configs = yaml.safe_load(file)

    args = merge_params(arg_params, configs)
    if args.CE == False:
        args.alpha = 0.
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

    testset = torchvision.datasets.ImageFolder(root=os.path.join(root, "val_8"),
                                               transform=transform_test)

    testclose_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                   shuffle=False, pin_memory=True, drop_last=False)

    inputencoder = InputEncoder(latent_dim=args.latent_dim)
    inputencoder = inputencoder.cuda()
    classencoder = ClassEncoder(latent_dim=args.latent_dim, img_size=128, num_classes=args.num_classes)
    classencoder = classencoder.cuda()
    decoder = Decoder(latent_dim=args.latent_dim, num_classes=args.num_classes)
    decoder = decoder.cuda()
    classifier = Classifier(latent_dim=args.latent_dim, num_classes=args.num_classes)
    classifier = classifier.cuda()

    # checkpoint = torch.load(os.path.join(args.load_path, 'closetrain/checkpoints/best.pth'))
    # inputencoder.load_state_dict(checkpoint['inputencoder_state_dict'])
    # classencoder.load_state_dict(checkpoint['classencoder_state_dict'])
    # decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer = optim.SGD([
        {'params': inputencoder.parameters()},
        {'params': classencoder.parameters()},
        {'params': decoder.parameters()}], lr=lr, momentum=momentum)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    if args.phase == 'test':
        checkpoint = torch.load(os.path.join(args.load_path, 'closetrain/checkpoints/best.pth'))
        inputencoder.load_state_dict(checkpoint['inputencoder_state_dict'])
        classencoder.load_state_dict(checkpoint['classencoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        # classifier.load_state_dict(checkpoint['classifier_state_dict'])
        _ = epoch_closeval(inputencoder, classencoder, classifier, testclose_loader, args)
        draw_fig(inputencoder, classencoder, decoder, testclose_loader, args)
        return

    best_KLacc = 0.
    for epoch in range(epochs):  # loop over the dataset multiple times
        _ = epoch_train(inputencoder, classencoder, decoder, classifier, trainloader, optimizer, epoch, args)
        close_KLacc = epoch_closeval(inputencoder, classencoder, classifier, testclose_loader, args)
        draw_fig(inputencoder, classencoder, decoder, testclose_loader, args)
        torch.save({'epoch': epoch,
                    'inputencoder_state_dict': inputencoder.state_dict(),
                    'classencoder_state_dict': classencoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict()},
                   os.path.join(args.save_path, 'checkpoints', 'latest.pth'))
        # if testcondisc_acc > best_condiscacc:
        #     best_condiscacc = testcondisc_acc
        if close_KLacc >= best_KLacc:
            best_KLacc = close_KLacc
            torch.save({'epoch': epoch,
                        'inputencoder_state_dict': inputencoder.state_dict(),
                        'classencoder_state_dict': classencoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'classifier_state_dict': classifier.state_dict()},
                       os.path.join(args.save_path, 'checkpoints', 'best.pth'))
        print("Best KL Acc: {:.3f}".format(best_KLacc))


if __name__ == "__main__":
    main()
